import heapq

from .planner import Planner
from .primitives import PathNode, PixelCoords
from .utils import (
    get_8_neighbors,
    reconstruct_path,
    euclidean_distance,
    check_collision_free,
)


class AStarImplementation(Planner):
    def preloop(self):
        self._counter = 0          # tiebreaker so heap never compares PathNode objects
        self.heap = []             # (f_cost, counter, node)
        self.open_set: set[PathNode] = set()
        self.g_costs: dict[PathNode, float] = {}
        self.h_costs: dict[PathNode, float] = {}
        self.node_registry: dict[tuple[int, int], PathNode] = {}
        self.visited_nodes: set[PathNode] = set()

        self.start_node = PathNode(coords=self.start_coords)
        self.goal_node = PathNode(coords=self.goal_coords)
        self.node_registry[self.start_node.coords.to_tuple()] = self.start_node
        self.node_registry[self.goal_node.coords.to_tuple()] = self.goal_node

        self.g_costs[self.start_node] = 0.0
        h = euclidean_distance(self.start_node, self.goal_node)
        self.h_costs[self.start_node] = h

        heapq.heappush(self.heap, (h, self._counter, self.start_node))
        self.open_set.add(self.start_node)
        self._counter += 1

    def step(self):
        if not self.heap:
            self.search_done = True
            return

        _, _, current_node = heapq.heappop(self.heap)

        # lazy deletion: stale heap entry for an already-visited node
        if current_node in self.visited_nodes:
            return

        self.open_set.discard(current_node)
        self.visited_nodes.add(current_node)

        if (
            euclidean_distance(current_node, self.goal_node) <= self.goal_threshold
            and check_collision_free(self.inflated_obstacle_map, current_node, self.goal_node)
        ):
            self.goal_node.parent = current_node
            self.goal_node.cost = (
                self.g_costs[current_node]
                + euclidean_distance(current_node, self.goal_node)
            )
            self.visited_nodes.add(self.goal_node)
            self.found_path = True
            self.search_done = True
            return

        neighbor_nodes = get_8_neighbors(
            current_node.coords, self.inflated_obstacle_map, self.node_registry
        )
        for n_node in neighbor_nodes:
            if n_node in self.visited_nodes:
                continue
            g_new = self.g_costs[current_node] + 1.0
            if n_node not in self.g_costs or g_new < self.g_costs[n_node]:
                n_node.parent = current_node
                n_node.g_cost = g_new
                self.g_costs[n_node] = g_new
                h = euclidean_distance(n_node, self.goal_node)
                self.h_costs[n_node] = h
                f = g_new + h
                heapq.heappush(self.heap, (f, self._counter, n_node))
                self.open_set.add(n_node)
                self._counter += 1

    def postloop(self):
        if self.goal_node.parent is not None:
            return reconstruct_path(self.goal_node), self.visited_nodes
        return [], self.visited_nodes
