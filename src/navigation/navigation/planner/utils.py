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

def world_2_occupancy_map(world_map:NDArray, start: PixelCoords, goal: PixelCoords):
    
    inflated_obstacles_map = inflated_obstacles(world_map, 5, start, goal)
    
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

def smooth_path_generation(path_coords, num_points=100):
    # 1. validate path length
    
    # 2. extract x and y
    # 3. compute path parameter t
    # 4. sample new t values
    # 5. spline x(t)
    # 6. spline y(t)
    # 7. rebuild smoothed coordinates
    # 8. return smoothed path
    pass

# ======================================================

# Visualization ====================================================== 
def draw_path():
    pass

def draw_target():
    pass
# ======================================================