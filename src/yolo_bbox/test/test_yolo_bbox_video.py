"""Unit tests for yolo_bbox_video helper functions."""
import queue
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Mock heavy dependencies before importing the module
sys.modules.setdefault('rclpy', MagicMock())
sys.modules.setdefault('rclpy.node', MagicMock())
sys.modules.setdefault('cv_bridge', MagicMock())
sys.modules.setdefault('ultralytics', MagicMock())
sys.modules.setdefault('cv2', MagicMock())
sys.modules.setdefault('sensor_msgs', MagicMock())
sys.modules.setdefault('sensor_msgs.msg', MagicMock())
sys.modules.setdefault('std_msgs', MagicMock())
sys.modules.setdefault('std_msgs.msg', MagicMock())

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'yolo_bbox'))

from yolo_bbox_video import _enqueue_latest, _select_device


class TestEnqueueLatest(unittest.TestCase):

    def test_puts_frame_into_empty_queue(self):
        q = queue.Queue(maxsize=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _enqueue_latest(q, frame)
        self.assertEqual(q.qsize(), 1)

    def test_drops_stale_frame_when_full(self):
        q = queue.Queue(maxsize=1)
        old_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        new_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        _enqueue_latest(q, old_frame)
        _enqueue_latest(q, new_frame)
        self.assertEqual(q.qsize(), 1)
        retrieved = q.get_nowait()
        np.testing.assert_array_equal(retrieved, new_frame)

    def test_queue_never_exceeds_maxsize(self):
        q = queue.Queue(maxsize=1)
        for i in range(10):
            frame = np.full((480, 640, 3), i, dtype=np.uint8)
            _enqueue_latest(q, frame)
        self.assertLessEqual(q.qsize(), 1)


class TestSelectDevice(unittest.TestCase):

    @patch('torch.cuda.is_available', return_value=True)
    def test_returns_cuda_when_available(self, _mock):
        self.assertEqual(_select_device('cuda'), 'cuda')

    @patch('torch.cuda.is_available', return_value=False)
    def test_falls_back_to_cpu_when_cuda_unavailable(self, _mock):
        self.assertEqual(_select_device('cuda'), 'cpu')

    @patch('torch.cuda.is_available', return_value=False)
    def test_cpu_request_stays_cpu(self, _mock):
        self.assertEqual(_select_device('cpu'), 'cpu')


if __name__ == '__main__':
    unittest.main()
