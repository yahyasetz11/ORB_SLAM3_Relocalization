"""Unit tests for yolo_bbox_video helper logic — no ROS2 or YOLO required."""
import queue
import unittest
from unittest.mock import patch
import numpy as np


def enqueue_latest(frame_queue: queue.Queue, frame: np.ndarray) -> None:
    """Drop stale frame and put the newest one. Mirrors image_callback logic."""
    try:
        frame_queue.put_nowait(frame)
    except queue.Full:
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put_nowait(frame)


def select_device(requested: str) -> str:
    """Return 'cpu' if CUDA requested but unavailable, else return requested."""
    import torch
    if requested == 'cuda' and not torch.cuda.is_available():
        return 'cpu'
    return requested


class TestEnqueueLatest(unittest.TestCase):

    def test_puts_frame_into_empty_queue(self):
        q = queue.Queue(maxsize=1)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        enqueue_latest(q, frame)
        self.assertEqual(q.qsize(), 1)

    def test_drops_stale_frame_when_full(self):
        q = queue.Queue(maxsize=1)
        old_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        new_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        enqueue_latest(q, old_frame)
        enqueue_latest(q, new_frame)          # should evict old, put new
        self.assertEqual(q.qsize(), 1)
        retrieved = q.get_nowait()
        np.testing.assert_array_equal(retrieved, new_frame)

    def test_queue_never_exceeds_maxsize(self):
        q = queue.Queue(maxsize=1)
        for i in range(10):
            frame = np.full((480, 640, 3), i, dtype=np.uint8)
            enqueue_latest(q, frame)
        self.assertLessEqual(q.qsize(), 1)


class TestSelectDevice(unittest.TestCase):

    @patch('torch.cuda.is_available', return_value=True)
    def test_returns_cuda_when_available(self, _mock):
        self.assertEqual(select_device('cuda'), 'cuda')

    @patch('torch.cuda.is_available', return_value=False)
    def test_falls_back_to_cpu_when_cuda_unavailable(self, _mock):
        self.assertEqual(select_device('cuda'), 'cpu')

    @patch('torch.cuda.is_available', return_value=False)
    def test_cpu_request_stays_cpu(self, _mock):
        self.assertEqual(select_device('cpu'), 'cpu')


if __name__ == '__main__':
    unittest.main()
