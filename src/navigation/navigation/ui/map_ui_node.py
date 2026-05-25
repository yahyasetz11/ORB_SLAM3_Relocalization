#!/usr/bin/env python3
# navigation/ui/map_ui_node.py

# ─── Standard library ────────────────────────────────────────────────────────
import os
import sys
import threading
from typing import Any, Dict, Optional

# ─── ROS core ────────────────────────────────────────────────────────────────
import rclpy
from rclpy.node import Node

# ─── ROS messages ────────────────────────────────────────────────────────────
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String

# ─── CvBridge ────────────────────────────────────────────────────────────────
from cv_bridge import CvBridge

# cv2 (loaded by cv_bridge) sets QT_QPA_PLATFORM_PLUGIN_PATH to its own
# bundled Qt plugins, which conflicts with PyQt5. Clear it now so Qt falls
# back to the system plugin path before PyQt5 is imported.
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

# ─── NumPy ───────────────────────────────────────────────────────────────────
from numpy.typing import NDArray

# ─── Ament package resource lookup ───────────────────────────────────────────
from ament_index_python.packages import get_package_share_directory

# ─── PyQt5 ───────────────────────────────────────────────────────────────────
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# ─── Navigation utilities ─────────────────────────────────────────────────────
from navigation.ui.ui_utils import (
    cv_to_pixmap,
    get_location_names,
    load_locations,
    search_location,
)


# ─────────────────────────────────────────────────────────────────────────────
# Clickable map label
# ─────────────────────────────────────────────────────────────────────────────

class ClickableMapLabel(QLabel):
    def __init__(self, on_click, parent=None):
        super().__init__(parent)
        self._on_click = on_click

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._on_click(event.x(), event.y(), self.width(), self.height())
        super().mousePressEvent(event)


# ─────────────────────────────────────────────────────────────────────────────
# Signal bridge
# ─────────────────────────────────────────────────────────────────────────────

class _SignalBus(QObject):
    image_signal  = pyqtSignal(object)
    status_signal = pyqtSignal(str, str)


# ─────────────────────────────────────────────────────────────────────────────
# Main node
# ─────────────────────────────────────────────────────────────────────────────

class MapUINode(Node):

    def __init__(self):
        super().__init__('map_ui_node')

        # ── Signal bridge ─────────────────────────────────────────────────────
        self._signals = _SignalBus()
        self.status_signal = self._signals.status_signal

        # ── Internal state ────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.current_img: Optional[NDArray] = None
        self.current_position: Optional[PointStamped] = None
        self._first_image    = True   # show one-shot status when first frame arrives
        self._first_position = True   # show one-shot status when first position arrives

        # ── Map directory ─────────────────────────────────────────────────────
        self.map_dir = os.path.join(
            get_package_share_directory('navigation'), 'maps'
        )
        # self.map_dir = os.path.join(
        #     os.path.expanduser("~"),
        #     "ORB_SLAM3_Relocalization",
        #     "src", "navigation", "navigation", "maps"
        # )

        # ── Map dimension params ──────────────────────────────────────────────
        self.declare_parameter('map_width_meters', 50.0)
        self.declare_parameter('map_height_meters', 50.0)
        self.map_width_meters  = self.get_parameter('map_width_meters').get_parameter_value().double_value
        self.map_height_meters = self.get_parameter('map_height_meters').get_parameter_value().double_value

        # ── ROS subscribers ───────────────────────────────────────────────────
        self.image_sub = self.create_subscription(
            Image, 'trails_image', self.image_callback, 10)
        
        self.pose_sub = self.create_subscription(
            PointStamped, 'current_position', self.position_callback, 10)

        # ── ROS publishers ────────────────────────────────────────────────────
        self.start_pub    = self.create_publisher(PointStamped, 'start_position', 10)
        self.goal_pub     = self.create_publisher(PointStamped, 'goal_position',  10)
        self.map_name_pub = self.create_publisher(String, '/map_name', 10)

        # ── Build UI ──────────────────────────────────────────────────────────
        self._init_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _init_ui(self):
        self._window = QMainWindow()
        self._window.setWindowTitle('Navigation Map')
        self._window.resize(1024, 768)
        self._window.setStyleSheet('background: #1e1e1e; color: white;')

        central = QWidget()
        self._window.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Top bar: map selector + manual load ───────────────────────────────
        top_bar = QHBoxLayout()

        self.map_combo = QComboBox()
        self.map_combo.setStyleSheet('padding: 4px; background: #2d2d2d; color: white;')
        self.map_combo.addItem('── Select map ──')
        if os.path.isdir(self.map_dir):
            for f in sorted(os.listdir(self.map_dir)):
                if f.endswith('.png'):
                    self.map_combo.addItem(f[:-4])
        self.map_combo.currentTextChanged.connect(self._on_map_selected)

        self.map_name_input = QLineEdit()
        self.map_name_input.setPlaceholderText('Map name…')
        self.map_name_input.setStyleSheet(
            'padding: 4px; background: #2d2d2d; color: white; border: 1px solid #555;'
        )
        self.map_name_input.returnPressed.connect(self._on_load_map)

        load_btn = QPushButton('Load')
        load_btn.setFixedWidth(50)
        load_btn.setStyleSheet('padding: 4px; background: #0078d4; color: white; border: none;')
        load_btn.clicked.connect(self._on_load_map)

        top_bar.addWidget(self.map_combo, stretch=2)
        top_bar.addWidget(self.map_name_input, stretch=3)
        top_bar.addWidget(load_btn)
        root.addLayout(top_bar)

        # ── Map display ───────────────────────────────────────────────────────
        self.map_label = ClickableMapLabel(self._on_map_click)
        self.map_label.setText('Waiting for map image…')
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.map_label.setScaledContents(True)
        self.map_label.setStyleSheet('background: #111;')
        root.addWidget(self.map_label)

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_label = QLabel('Ready')
        self.status_label.setFixedHeight(24)
        self.status_label.setStyleSheet('color: white; padding: 2px 6px; background: #2d2d2d;')
        root.addWidget(self.status_label)

        # ── Connect ROS → Qt signals ──────────────────────────────────────────
        self._signals.image_signal.connect(self._update_display)
        self._signals.status_signal.connect(self.show_status)

        self._window.show()

        self.show_status('[INIT] Ready — select or load a map', 'info')

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def image_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.current_img = img
        if self._first_image:
            h, w = img.shape[:2]
            self._signals.status_signal.emit(
                f'[MAP] First frame received  {w}×{h} px', 'success'
            )
            self._first_image = False
        self._signals.image_signal.emit(img)

    def position_callback(self, msg: PointStamped):
        self.current_position = msg
        if self._first_position:
            self._signals.status_signal.emit(
                f'[POS] First position  ({msg.point.x:.2f}, {msg.point.y:.2f})', 'success'
            )
            self._first_position = False

    # ── Qt slots ──────────────────────────────────────────────────────────────

    def _update_display(self, img: Optional[NDArray] = None):
        base = img if img is not None else self.current_img
        
        if base is None:
            return
        
        self.map_label.setPixmap(cv_to_pixmap(base))

    def _on_map_selected(self, name: str):
        if name.startswith('──'):
            return
        name_msg = String()
        name_msg.data = name
        self.map_name_pub.publish(name_msg)
        self.show_status(f'[MAP] Loading "{name}"…', 'info')

    def _on_load_map(self):
        name = self.map_name_input.text().strip()
        if not name:
            return
        png_exists = os.path.isfile(os.path.join(self.map_dir, f'{name}.png'))
        csv_exists = os.path.isfile(os.path.join(self.map_dir, f'{name}.csv'))
        if not png_exists and not csv_exists:
            self.show_status(f'[MAP] Map "{name}" not found — no .png or .csv', 'error')
            return
        name_msg = String()
        name_msg.data = name
        self.map_name_pub.publish(name_msg)
        if not png_exists:
            self.show_status(f'[MAP] Building "{name}" from CSV…', 'info')
        else:
            self.show_status(f'[MAP] Loading "{name}"…', 'info')

    def _on_map_click(self, px, py, img_w, img_h):
        if img_w == 0 or img_h == 0:
            return
        if self.current_img is None:
            self.show_status('[GOAL] No map loaded yet', 'warn')
            return
        map_h, map_w = self.current_img.shape[:2]
        # px is horizontal (column ↔), py is vertical from top (row ↓)
        col = int((px / img_w) * map_w)
        row = int((py / img_h) * map_h)
        # publish in user coords: origin bottom-left, X upward, Y rightward
        user_x = map_h - 1 - row
        user_y = col
        now = self.get_clock().now().to_msg()
        goal_msg = PointStamped()
        goal_msg.header.stamp    = now
        goal_msg.header.frame_id = 'map'
        goal_msg.point.x         = float(user_x)
        goal_msg.point.y         = float(user_y)
        self.goal_pub.publish(goal_msg)
        self.show_status(f'[GOAL] Set goal (x={user_x}, y={user_y})', 'success')

    # ── Status bar ────────────────────────────────────────────────────────────

    def show_status(self, message: str, level: str = 'info'):
        color_map = {
            'info':    'white',
            'success': '#4CAF50',
            'warn':    '#FFA726',
            'error':   '#F44336',
        }
        color = color_map.get(level, 'white')
        self.status_label.setStyleSheet(
            f'color: {color}; padding: 2px 6px; background: #2d2d2d;'
        )
        self.status_label.setText(message)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    node = MapUINode()

    ros_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    ros_thread.start()

    try:
        app.exec()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
