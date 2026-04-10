from dataclasses import dataclass, field

class PixelCoords():
    def __init__(self, x:int, y:int):
        self.x_coords = round(x)
        self.y_coords = round(y)
    
    @property
    def x(self):
        return self.x_coords
    @property
    def y(self):
        return self.y_coords
    
    def __eq__(self, value):
        if isinstance(value, PixelCoords):
            return self.x == value.x and self.y == value.y
        return False
    def __hash__(self):
        return hash((self.x, self.y))
    
    def to_tuple(self):
        return (self.x, self.y)
        
@dataclass
class PathNode:
    coords: PixelCoords
    parent: "PathNode"=None
    g_cost: float = 0.0
    h_cost: float = 0.0
    cost: float = 0.0 

    def __post_init__(self) -> None:
        pass

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost
    
    def __lt__(self, other: "PathNode") -> bool:
        if self.f_cost < other.f_cost:
            return False
        return True

    def has_parent(self) -> bool:
        if self.parent is not None:
            return True
        return False
    
    def __eq__(self, other):
        if isinstance(other, PathNode):
            return self.coords == other.coords
        return False
    
    def __hash__(self):
        return hash(self.coords)