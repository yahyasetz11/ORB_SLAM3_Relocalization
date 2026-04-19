from abc import abstractmethod
from typing import final

from numpy.typing import NDArray

from .primitives import PathNode, PixelCoords
from .utils import world_2_occupancy_map, is_standable, is_in_bounds

class Planner():
    def __init__(
        self,
        world_map:NDArray,
        start_coords:PixelCoords, goal_coords:PixelCoords,
        goal_threshold:float=3, iter_limit:int=10000,
        inflation_radius:int=5,
        ):
        
        self.world_map = world_map.copy()
        self.start_coords = start_coords
        self.goal_coords = goal_coords
        self.goal_threshold = goal_threshold
        self.iter_limit = iter_limit
        self.found_path = False
        self.search_done = False
        self.occupancy_map = None
        self.inflated_obstacle_map = None
        self.inflation_radius = inflation_radius

    def set_map(self):
        self.occupancy_map, self.inflated_obstacle_map = world_2_occupancy_map(self.world_map, self.start_coords, self.goal_coords, self.inflation_radius)

    def normalize_start(self):
        if not isinstance(self.start_coords, PixelCoords):
            self.start_coords = PixelCoords(*self.start_coords)

    def normalize_goal(self):
        if not isinstance(self.goal_coords, PixelCoords):
            self.goal_coords = PixelCoords(*self.goal_coords)

    def is_ready(self):
        if self.occupancy_map is None:
            raise ValueError("Map not ready")
        if self.start_coords is None:
            raise ValueError("Start coords not ready")
        if self.goal_coords is None:
            raise ValueError("Goal coords not ready")
        return True

    def validate_request(self):
        if not self.is_ready():
            raise ValueError("Planner is not ready: map, start, or goal is missing.")

        if not is_in_bounds(self.start_coords, self.occupancy_map):
            raise ValueError(f"Start out of bounds: ({self.start_coords.x}, {self.start_coords.y})")

        if not is_in_bounds(self.goal_coords, self.occupancy_map):
            raise ValueError(f"Goal out of bounds: ({self.goal_coords.x}, {self.goal_coords.y})")

        if not is_standable(self.start_coords, self.occupancy_map):
            raise ValueError(f"Start is blocked: ({self.start_coords.x}, {self.start_coords.y})")

        if not is_standable(self.goal_coords, self.occupancy_map):
            raise ValueError(f"Goal is blocked: ({self.goal_coords.x}, {self.goal_coords.y})")

        return True
    
    @final
    def plan(self):
        self.set_map()
        self.normalize_goal()
        self.normalize_start()
        self.found_path = False
        self.search_done = False
        
        if not self.validate_request():
            raise ValueError("Not ready")

        self.preloop()

        for i in range(self.iter_limit):
            self.step()
            if self.found_path:
                break
            if self.search_done:
                break

        path, visited_nodes = self.postloop()
        return path, visited_nodes
            
        
    @abstractmethod
    def preloop(self) -> None:
        ...
    @abstractmethod
    def step(self) -> None:
        ...
    @abstractmethod
    def postloop(self) -> tuple[list[PathNode], set[PathNode]]:
        ...