# navigation/navigation/planner/a_star.py
import cv2
import numpy as np

from .planner import Planner
from .primitives import PathNode, PixelCoords
from .utils import (
    get_8_neighbors,
    reconstruct_path,
    manhattan_distance,
    euclidean_distance,
    check_collision_free,
)

class AStarImplementation(Planner):
    def preloop(self):
        # This is for illustrative purposes only, feel free to modify
        self.queue: set[PathNode] = set()
        self.g_costs: dict[PathNode, float] = {}
        self.h_costs: dict[PathNode, float] = {}
        self.node_registry: dict[tuple[int, int], PathNode] = {}
        self.visited_nodes: set[PathNode] = set()

        self.start_node = PathNode(coords=self.start_coords)
        self.goal_node = PathNode(coords=self.goal_coords)

        self.queue.add(self.start_node)
        self.node_registry[self.start_node.coords.to_tuple()] = self.start_node
        self.node_registry[self.goal_node.coords.to_tuple()] = self.goal_node

        self.g_costs[self.start_node] = 0.0
        self.h_costs[self.start_node] = euclidean_distance(
            self.start_node,
            self.goal_node
        )

        # ===================
        # self.current_node = self.start_node

    def step(self):
        # ===== some given data/parameters =====
        self.start_node 
        self.goal_node
        self.world_map # bgr
        self.occupancy_map # bool
        self.inflated_obstacle_map
        self.goal_threshold
        # self.grid_size # for sampling neighbor nodes
        # ==========
        # to sample neighbor nodes, use 
        # self.get_neighbor_nodes(current_node)

        # ====================
        # 1. Check queue is empty, if yes, terminate loop
        if not self.queue: 
            self.search_done = True # only set this on termination
            return
        
        # 2. Choose the lowest cost node from the queue
        lowest_cost = float('inf')
        for nodes in self.queue:
            cost = self.g_costs[nodes]+self.h_costs[nodes]
            if cost < lowest_cost:
                lowest_cost = cost
                current_node = nodes
        
        # 3. Remove node from the queue
        self.queue.remove(current_node)

        # 4. Update visited nodes
        self.visited_nodes.add(current_node)

        # 5. Check if goal is reached (with self.goal_threshold)
        if euclidean_distance(current_node, self.goal_node) <= self.goal_threshold and check_collision_free(self.inflated_obstacle_map, current_node, self.goal_node):
            # Goal is reached, so...
            # a. Set the goal's parent node as current node
            self.goal_node.parent = current_node
            self.goal_node.cost = self.g_costs[current_node] + euclidean_distance(current_node, self.goal_node)
            self.visited_nodes.add(self.goal_node)
            self.found_path = True 
            self.search_done = True # only set this on termination
            return
        
        # 6. Get valid neighbor nodes into the queue
        # We also need to validate the cost (g and h) and set up parent
        neighbor_nodes = get_8_neighbors(current_node.coords, self.inflated_obstacle_map, self.node_registry)
        for n_nodes in neighbor_nodes:
            g_new = self.g_costs[current_node] + euclidean_distance(current_node, n_nodes)
            # Update the new g value if there is a new g value 
            if n_nodes not in self.g_costs or g_new < self.g_costs[n_nodes]: 
                # Update cost/parent/g/h
                n_nodes.parent = current_node
                n_nodes.g_cost = g_new
                self.g_costs[n_nodes] = g_new
                self.h_costs[n_nodes] = euclidean_distance(n_nodes, self.goal_node)
                self.queue.add(n_nodes)

        

    def postloop(self):
        if self.goal_node.parent is not None:
            path_nodes = reconstruct_path(self.goal_node)
            return (
                path_nodes, 
                # set() # replace with set of visited nodes
                self.visited_nodes
            )
        else:
            return (
                [],
                self.visited_nodes
            )