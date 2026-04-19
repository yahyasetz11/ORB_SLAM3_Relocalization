from typing import List, Optional

import cv2
import numpy as np

from numpy.typing import NDArray
from navigation.planner.primitives import PixelCoords, PathNode

# Map checking functions ======================================================

def bresenham(
    x0:int, 
    x1:int, 
    y0:int, 
    y1:int
) -> list[tuple[int, int]]:
    rec = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            rec.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            rec.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return rec

def is_in_bounds(coords: PixelCoords, occupancy_grid:NDArray) -> bool:
    grid_height, grid_width = occupancy_grid.shape[:2]
    
    if coords.x >= grid_width or coords.y >= grid_height: 
        return False
    
    if coords.x < 0 or coords.y < 0:
        return False
    
    return True


def is_standable(coords: PixelCoords, occupancy_grid) -> bool:
    if occupancy_grid[coords.y, coords.x]:
        return False
    return True

def check_collision_free(
    occupancy_map:NDArray, 
    source_node:PathNode, 
    target_node:PathNode
) -> bool:
    if (
        not is_in_bounds(source_node.coords, occupancy_map) or
        not is_in_bounds(target_node.coords, occupancy_map)
    ):
        raise RuntimeError()
    source_coordinates = source_node.coords
    target_coordinates = target_node.coords
    line_pixels = bresenham(
        x0=source_coordinates.x_coords, 
        x1=target_coordinates.x_coords, 
        y0=source_coordinates.y_coords, 
        y1=target_coordinates.y_coords
    )
    for pixel in line_pixels:
        if occupancy_map[pixel[1], pixel[0]] == 1:
            return False
    return True

def get_8_neighbors(
    coords: PixelCoords, 
    occupancy_grid, 
    node_registry: dict[tuple[int, int], PathNode]
) -> List[PathNode]:
    x_coords = coords.x_coords
    y_coords = coords.y_coords
    
    current_key = coords.to_tuple()
    
    
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),     
        (1, 1), (1, -1), (-1, 1), (-1, -1)    
    ]

    result = []
    
    for xx, yy in directions:
        coordinates = PixelCoords(x_coords + xx, y_coords + yy)
        
        if not is_in_bounds(coordinates, occupancy_grid) or not is_standable(coordinates, occupancy_grid):
            continue
        
        key = coordinates.to_tuple()
        
        if key not in node_registry:
            node_registry[key] = PathNode(coordinates)
            
        if not check_collision_free(occupancy_grid, node_registry[current_key], node_registry[key]):
            continue
          
        result.append(node_registry[key])
        
    return result

# ======================================================

# Coordinate conversion ======================================================

def world_to_pixel(wx: float, wy: float, map_info) -> PixelCoords:
    res = map_info.resolution
    ox  = map_info.origin.position.x
    oy  = map_info.origin.position.y
    return PixelCoords(int((wx - ox) / res), int((wy - oy) / res))

def pixel_to_world(px: int, py: int, map_info) -> tuple[float, float]:
    res = map_info.resolution
    ox  = map_info.origin.position.x
    oy  = map_info.origin.position.y
    return ox + px * res, oy + py * res

# ======================================================

# Distance calculation ======================================================

def manhattan_distance(start: PathNode, goal: PathNode) -> float:
    return np.abs(start.coords.x_coords - goal.coords.x_coords) + np.abs(start.coords.y_coords - goal.coords.y_coords)

def euclidean_distance(start: PathNode, goal: PathNode) -> float:
    return np.sqrt((start.coords.x_coords - goal.coords.x_coords)**2 + (start.coords.y_coords - goal.coords.y_coords)**2)

# ======================================================

# World transform ======================================================

def inflated_obstacles(occupancy_map:NDArray, inflation_radius:int, start: PixelCoords, goal:PixelCoords):
    directions = [
        (0, 0),
        (0, 1), (1, 0), (0, -1), (-1, 0),     
        (1, 1), (1, -1), (-1, 1), (-1, -1)    
    ]
    grid_height, grid_width = occupancy_map.shape[:2]
    new_map = occupancy_map.copy()
    
    for r in range(1, inflation_radius):
        scaled_directions = [(dx * r, dy * r) for dx, dy in directions]
        
        for yy in range(grid_height):
            for xx in range(grid_width):
                if occupancy_map[yy, xx] == 1:
                    for dx, dy in scaled_directions:
                    
                        coordinates = PixelCoords(xx+dx, yy+dy)
                        if coordinates != start and coordinates != goal:
                            if is_in_bounds(coordinates, occupancy_map):
                                new_map[yy+dy, xx+dx] = 1
                                
    return new_map  

def world_2_occupancy_map(world_map:NDArray, start: PixelCoords, goal: PixelCoords, inflation_radius:int=5):

    inflated_obstacles_map = inflated_obstacles(world_map, inflation_radius, start, goal)

    return world_map, inflated_obstacles_map

# ======================================================

# Path generation ======================================================

def reconstruct_path(goal_node: PathNode) -> List[PathNode]:
    path_list = []
    current_node = goal_node
    while current_node:
        path_list.append(current_node)
        current_node = current_node.parent
        
    return path_list[::-1] 

def natural_cubic_spline(t, value):
    if len(t) != len(value):
        raise ValueError("t and value must have the same length")
    if len(t) < 2:
        raise ValueError("Need at least two points")
    if any(t[i+1] <= t[i] for i in range(len(t)-1)):
        raise ValueError("t must be strictly increasing")
    
    n = len(t) - 1
    
    h = [0.0] * n
    alpha = [0.0] * (n + 1)
    l = [0.0] * (n + 1)
    mu = [0.0] * (n + 1)
    z = [0.0] * (n + 1)
    a = value[:]              # a[i] = value[i]
    b = [0.0] * n
    c = [0.0] * (n + 1)
    d = [0.0] * n
    # step 1: compute h[i]
    for i in range(n):
        h[i] = t[i+1] - t[i]
        
    # step 2: compute alpha[i]
    for i in range(1, n):
        alpha[i] = 3*((a[i+1] - a[i])/h[i]) - 3*((a[i] - a[i-1])/h[i-1])

    # step 3: forward pass, l, mu, z arrays
    # Initial
    l[0] = 1.0
    mu[0] = 0.0
    z[0] = 0.0
    
    for i in range(1, n):
        l[i] = 2 * (t[i+1] - t[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]
    
    # Ending 
    l[n] = 1.0
    z[n] = 0.0
    c[n] = 0.0
    # step 4: backward pass
    #   compute a, b, c, d
    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (a[j+1] - a[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])
    
    return a, b, c, d

def compute_turning_angle(prev:PixelCoords, curr:PixelCoords, next:PixelCoords):
    v1 = np.array([(curr.x - prev.x), (curr.y - prev.y)])
    v2 = np.array([(next.x - curr.x), (next.y - curr.y)])
    
    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    
    if l1 == 0 or l2 == 0:
        return 0.0 
    
    dot = np.dot(v1, v2)
    dot = np.degrees(np.arccos(np.clip(dot / (l1 * l2), -1.0, 1.0)))
    
    return dot
    
def turn_region(path_coords:List[PathNode], threshold):
    
    turn_indices = set()
    
    for i, node in enumerate(path_coords):
        if i == 0 or i == len(path_coords) - 1:
            continue
        
        angle = compute_turning_angle(
            path_coords[i-1].coords,
            path_coords[i].coords,
            path_coords[i+1].coords
            )
        
        # Straight 0, Turn 1
        if angle > threshold:
            turn_indices.add(i)
    
    return turn_indices

def expand_turn_regions(turn_indices, path_length, radius):
    # Check if it's a turn
    expand = set()
    for idx in turn_indices:
        for j in range(max(0, idx - radius), min(path_length, idx + radius)):
            expand.add(j)
    
    return expand

def reduce_straight_regions(turn_indices, path_length, interval):
    straight = set()
    straight.add(0)
    straight.add(path_length - 1)
    # Everything not a straight is consider straight
    for i in range(path_length):
        if i in turn_indices:
            continue
        if i == 0 or i == path_length - 1:
            continue
        if i % interval == 0:
            straight.add(i)
            
    return straight


def generate_adaptive_waypoints(path_coords,
                                straight_interval=20,
                                turn_threshold=15.0,
                                turn_density=5):
    length = len(path_coords)
    turn_indices = turn_region(path_coords, turn_threshold)
    turn_indices = expand_turn_regions(turn_indices, length, turn_density)
    straight_indices = reduce_straight_regions(turn_indices, length, straight_interval)
    smooth_indices = turn_indices | straight_indices
    
    return [path_coords[i] for i in sorted(smooth_indices)]

def smooth_path_generation(
    path_coords: List[PathNode],
    straight_interval=20,
    turn_density=5,
    num_points: int = 300,
) -> List[PathNode]:
    # 1. validate path length
    if len(path_coords) < 4:
        return []

    # 2. downsample to sparse control points so the spline curves freely
    #    instead of being forced through every grid pixel
    # indices = np.linspace(0, len(path_coords) - 1, num_control_points, dtype=int)
    # control = [path_coords[i] for i in indices]
    control = generate_adaptive_waypoints(path_coords, straight_interval, turn_density)
    
    if len(control) < 4:
        return []

    # 3. extract x and y from control points
    xs = [node.coords.x_coords for node in control]
    ys = [node.coords.y_coords for node in control]

    # 4. compute parameter t as cumulative arc length over control points
    t = [0.0]
    for i in range(1, len(xs)):
        dist = np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2)
        t.append(t[-1] + dist)

    # deduplicate coincident points that would break strict-increasing requirement
    unique = [0]
    for i in range(1, len(t)):
        if t[i] > t[unique[-1]]:
            unique.append(i)
    if len(unique) < 4:
        return []
    t  = [t[i]  for i in unique]
    xs = [xs[i] for i in unique]
    ys = [ys[i] for i in unique]

    # 5. sample densely along the full arc length
    t_sampled = np.linspace(t[0], t[-1], num_points).tolist()

    # 6. fit splines
    ax, bx, cx, dx = natural_cubic_spline(t, xs)
    ay, by, cy, dy = natural_cubic_spline(t, ys)

    def _eval(a, b, c, d, t_knots, tq):
        n = len(t_knots) - 1
        i = min(n - 1, max(0, int(np.searchsorted(t_knots, tq, side='right')) - 1))
        dt = tq - t_knots[i]
        return a[i] + b[i]*dt + c[i]*dt**2 + d[i]*dt**3

    # 7. rebuild smoothed path
    smoothed = []
    for tq in t_sampled:
        x_s = _eval(ax, bx, cx, dx, t, tq)
        y_s = _eval(ay, by, cy, dy, t, tq)
        smoothed.append(PathNode(PixelCoords(x_s, y_s)))

    return smoothed

# ======================================================

# Visualization ======================================================

def draw_path(
    img: NDArray,
    path: List[PathNode],
    show_path: bool = True,
    path_color: tuple = (60, 179, 113),
    outline_color: tuple = (255, 255, 255),
    path_thickness: int = 4,
    outline_thickness: int = 7,
    arrow_color: tuple = (0, 165, 255),
    arrow_interval: int = 30,
    arrow: bool = True,
) -> NDArray:
    if not path:
        return img

    out = img.copy()

    pts = np.array(
        [[int(n.coords.x_coords), int(n.coords.y_coords)] for n in path],
        dtype=np.int32
    ).reshape(-1, 1, 2)

    if show_path:
        # white outline first, then colored line on top (Google Maps style)
        cv2.polylines(out, [pts], isClosed=False, color=outline_color, thickness=outline_thickness, lineType=cv2.LINE_AA)
        cv2.polylines(out, [pts], isClosed=False, color=path_color,    thickness=path_thickness,    lineType=cv2.LINE_AA)

    if arrow:
        for i in range(0, len(path) - 1, arrow_interval):
            j = min(i + arrow_interval, len(path) - 1)
            src = (int(path[i].coords.x_coords), int(path[i].coords.y_coords))
            dst = (int(path[j].coords.x_coords), int(path[j].coords.y_coords))
            cv2.arrowedLine(out, src, dst, arrow_color, thickness=2, tipLength=0.3, line_type=cv2.LINE_AA)

    return out


def draw_target(
    img: NDArray,
    coords: PixelCoords,
    color: tuple = (0, 0, 255),
    radius: int = 5,
) -> NDArray:
    out = img.copy()
    center = (coords.x_coords, coords.y_coords)
    cv2.circle(out, center, radius, color, thickness=-1)
    cv2.circle(out, center, radius + 2, color, thickness=1)
    return out

# ======================================================