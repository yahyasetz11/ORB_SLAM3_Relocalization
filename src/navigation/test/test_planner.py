import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from navigation.planner.primitives import PixelCoords, PathNode
from navigation.planner.utils import inflated_obstacles
from navigation.planner.a_star import AStarImplementation


def make_node(x, y, g=0.0, h=0.0):
    node = PathNode(coords=PixelCoords(x, y))
    node.g_cost = g
    node.h_cost = h
    return node


class TestPathNodeLt:
    def test_lower_fcost_is_less_than(self):
        cheap = make_node(0, 0, g=1.0, h=1.0)   # f=2.0
        expensive = make_node(1, 1, g=3.0, h=3.0)  # f=6.0
        assert cheap < expensive

    def test_higher_fcost_is_not_less_than(self):
        cheap = make_node(0, 0, g=1.0, h=1.0)   # f=2.0
        expensive = make_node(1, 1, g=3.0, h=3.0)  # f=6.0
        assert not (expensive < cheap)


class TestInflatedObstacles:
    def _make_map(self, h=20, w=20):
        m = np.zeros((h, w), dtype=np.uint8)
        m[10, 10] = 1   # single obstacle cell
        return m

    def test_adjacent_cells_are_inflated(self):
        m = self._make_map()
        result = inflated_obstacles(m, inflation_radius=3)
        # cells near (10,10) must be obstacles
        assert result[10, 11] == 1
        assert result[11, 10] == 1

    def test_original_obstacle_stays(self):
        m = self._make_map()
        result = inflated_obstacles(m, inflation_radius=3)
        assert result[10, 10] == 1

    def test_far_cells_are_free(self):
        m = self._make_map()
        result = inflated_obstacles(m, inflation_radius=3)
        assert result[0, 0] == 0

    def test_start_goal_cleared_when_provided(self):
        m = self._make_map()
        # place start right next to obstacle — inflation would normally block it
        start = PixelCoords(11, 10)
        goal  = PixelCoords(0, 0)
        result = inflated_obstacles(m, inflation_radius=3, start=start, goal=goal)
        assert result[start.y_coords, start.x_coords] == 0
        assert result[goal.y_coords, goal.x_coords] == 0

    def test_no_start_goal_does_not_crash(self):
        m = self._make_map()
        result = inflated_obstacles(m, inflation_radius=3)
        assert result is not None

    def test_diagonal_corner_cells_are_inflated(self):
        m = self._make_map()   # obstacle at (10,10)
        result = inflated_obstacles(m, inflation_radius=3)
        # Original algorithm marks (10±2, 10±2); MORPH_RECT does too; MORPH_ELLIPSE does not
        assert result[8, 8] == 1    # (col=8, row=8) i.e. x=8,y=8
        assert result[12, 12] == 1
        assert result[8, 12] == 1
        assert result[12, 8] == 1


class TestPlannerCachedInflation:
    def _simple_map(self):
        # 20x20 free map (all zeros)
        return np.zeros((20, 20), dtype=np.uint8)

    def test_planner_accepts_inflated_map(self):
        world_map = self._simple_map()
        pre_inflated = world_map.copy()  # already-computed inflated map
        start = PixelCoords(1, 1)
        goal  = PixelCoords(18, 18)
        planner = AStarImplementation(
            world_map=world_map,
            start_coords=start,
            goal_coords=goal,
            iter_limit=10000,
            inflated_map=pre_inflated,
        )
        path, _ = planner.plan()
        assert len(path) > 0, "Expected a path on a free map"
