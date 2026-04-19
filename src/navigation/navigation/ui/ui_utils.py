import os
# Image processing
import cv2
import numpy as np

# json
import json

# PyQt5 - image conversion
from PyQt5.QtGui import QImage, QPixmap

# ROS message types - just for type hints and unpacking
from nav_msgs.msg import Path
from nav_msgs.msg import OccupancyGrid        # for map metadata
from geometry_msgs.msg import PoseStamped

# YAML - for loading locations file
import yaml

# Typing - optional but helpful
from typing import Optional, Tuple, Dict, List

# NumPy typing
from numpy.typing import NDArray

# Planner primitives - for type hints in draw_path / draw_marker
from navigation.planner.primitives import PathNode, PixelCoords

# Image conversion ======================================================
def cv_to_pixmap(img: NDArray) -> QPixmap:
    h, w, ch = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

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

def draw_marker(
    img: NDArray,
    coords: PixelCoords,
    name: str,
    radius: int = 10,
    outline_color: tuple = (255, 255, 255),
    outline_radius: int = 13,
) -> NDArray:
    if name == "start":
        color = (60, 179, 113)   # green  (matches path color)
    elif name == "goal":
        color = (0, 0, 220)      # red
    else:
        raise ValueError("start/goal point spelling error")

    out = img.copy()
    cx = int(coords.x_coords)
    cy = int(coords.y_coords)

    # white outline first, then colored fill on top (Google Maps style)
    cv2.circle(out, (cx, cy), outline_radius, outline_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(out, (cx, cy), radius,         color,         -1, lineType=cv2.LINE_AA)

    return out

# Location loading ======================================================

def load_locations(json_path):
    """
    json file structure:
    {
    "Lab A": {
        "x": 1.5,
        "y": 2.3,
        "yaw": 0.0,
        "description": "Main laboratory"
    },
    "Entrance": {
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
        "description": "Main entrance"
    }
}
    """
    # Open JSON
    if not os.path.exists(json_path):
        raise ValueError(f"Landmark file does not exist: {json_path}")
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in landmark file: {e}")
  
def get_location_names(data: dict) -> list:
    return sorted(data.keys())

def search_location(data:dict, query):
    query = query.strip().lower()
    # Exact match
    for key in data:
        if query == key.strip().lower():
            return (key, data[key])
    # Starting with
    for key in data:
        if key.strip().lower().startswith(query):
            return (key, data[key])
    # Contains
    for key in data:
        if query in key.strip().lower():
            return (key, data[key])
    
    return None