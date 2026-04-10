# navigation/navigation/main.py

import numpy as np
import cv2
from typing import List
from navigation.planner.a_star import AStarImplementation
from navigation.planner.primitives import PixelCoords, PathNode


def create_test_map(width=100, height=100):
    """
    0 = free hallway
    1 = obstacle
    """
    grid = np.ones((height, width), dtype=np.uint8)

    cx = width // 2

    loop_w = 48
    loop_h = 32
    hall_thickness = 20      # wider outer hallway
    middle_thickness = 30    # wider center connector
    gap_between_loops = 4

    top_y = 10
    bottom_y = top_y + loop_h + gap_between_loops

    left_x = cx - loop_w // 2
    right_x = cx + loop_w // 2

    def carve_rect(y1, y2, x1, x2):
        grid[y1:y2, x1:x2] = 0

    # Top loop outer hallway
    carve_rect(top_y, top_y + hall_thickness, left_x, right_x)                        # top
    carve_rect(top_y + loop_h - hall_thickness, top_y + loop_h, left_x, right_x)      # bottom
    carve_rect(top_y, top_y + loop_h, left_x, left_x + hall_thickness)                # left
    carve_rect(top_y, top_y + loop_h, right_x - hall_thickness, right_x)              # right

    # Top inner obstacle
    inner_w = 12
    inner_h = 12
    grid[
        top_y + loop_h // 2 - inner_h // 2 : top_y + loop_h // 2 + inner_h // 2,
        cx - inner_w // 2 : cx + inner_w // 2
    ] = 1

    # Bottom loop outer hallway
    carve_rect(bottom_y, bottom_y + hall_thickness, left_x, right_x)                  # top
    carve_rect(bottom_y + loop_h - hall_thickness, bottom_y + loop_h, left_x, right_x)# bottom
    carve_rect(bottom_y, bottom_y + loop_h, left_x, left_x + hall_thickness)          # left
    carve_rect(bottom_y, bottom_y + loop_h, right_x - hall_thickness, right_x)        # right

    # Bottom inner obstacle
    grid[
        bottom_y + loop_h // 2 - inner_h // 2 : bottom_y + loop_h // 2 + inner_h // 2,
        cx - inner_w // 2 : cx + inner_w // 2
    ] = 1

    # Wider middle connector
    connector_y1 = top_y + loop_h - hall_thickness // 2
    connector_y2 = bottom_y + hall_thickness // 2
    carve_rect(
        connector_y1,
        connector_y2,
        cx - middle_thickness // 2,
        cx + middle_thickness // 2
    )

    return grid

def visualize(grid, path:List[PathNode], start:PixelCoords, goal:PixelCoords):
    """
    Simple visualization using OpenCV
    """
    img = np.stack([grid * 255]*3, axis=-1)  # convert to RGB

    # draw path
    for node in path:
        x, y = node.coords.x_coords, node.coords.y_coords
        img[y, x] = (0, 255, 0)

    # draw start
    img[start.y_coords, start.x_coords] = (255, 0, 0)

    # draw goal
    img[goal.y_coords, goal.x_coords] = (0, 0, 255)

    cv2.imshow("Path", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def main():
    # 1. Create or load map
    occupancy_map = create_test_map()

    # 2. Define start and goal
    start = PixelCoords(40, 22)
    goal = PixelCoords(60, 70)
    
    """
    fixed_pairs = [
        # top loop left -> top loop right
        (PixelCoords(34, 16), PixelCoords(66, 16)),

        # top loop -> bottom loop through middle connector
        (PixelCoords(34, 20), PixelCoords(66, 78)),

        # bottom loop left -> bottom loop right
        (PixelCoords(34, 84), PixelCoords(66, 84)),

        # top right -> bottom left
        (PixelCoords(66, 30), PixelCoords(34, 70)),
    ]
    """
    
    # 3. Create planner
    planner = AStarImplementation(
        world_map=occupancy_map,
        start_coords=start,
        goal_coords=goal,
        iter_limit=10000
    )

    # 4. Run planner
    try:
        path, visited = planner.plan()
    except ValueError as e:
        print("Planning failed:", e)
        return

    # 5. Print result
    print(f"Path length: {len(path)}")
    print(f"Visited nodes: {len(visited)}")

    # 6. Visualize (optional)
    visualize(occupancy_map, path, start, goal)

if __name__ == "__main__":
    main()