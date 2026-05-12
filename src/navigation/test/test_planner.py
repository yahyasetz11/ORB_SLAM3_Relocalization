import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from navigation.planner.primitives import PixelCoords, PathNode


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
