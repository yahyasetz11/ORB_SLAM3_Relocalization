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
        self.locations: Dict[str, Any] = {}
        self._first_image    = True   # show one-shot status when first frame arrives
        self._first_position = True   # show one-shot status when first position arrives

        # ── Load named locations ──────────────────────────────────────────────
        try:
            pkg_share = get_package_share_directory('navigation')
            locations_file = os.path.join(pkg_share, 'config', 'locations.yaml')
            self.locations = load_locations(locations_file)
            self.get_logger().info(
                f'Loaded {len(self.locations)} location(s) from {locations_file}'
            )
        except Exception as exc:
            self.get_logger().warn(f'Could not load locations.yaml: {exc}')

        # ── ROS subscribers ───────────────────────────────────────────────────
        self.image_sub = self.create_subscription(
            Image, 'trails_image', self.image_callback, 10)
        
        self.pose_sub = self.create_subscription(
            PointStamped, 'current_position', self.position_callback, 10)

        # ── ROS publishers ────────────────────────────────────────────────────
        self.start_pub = self.create_publisher(PointStamped, 'start_position', 10)
        self.goal_pub  = self.create_publisher(PointStamped, 'goal_position',  10)

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

        # ── Top bar: search + go + dropdown ──────────────────────────────────
        top_bar = QHBoxLayout()

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText('Search location…')
        self.search_box.setStyleSheet(
            'padding: 4px; background: #2d2d2d; color: white; border: 1px solid #555;'
        )
        self.search_box.returnPressed.connect(self._on_search)

        go_btn = QPushButton('Go')
        go_btn.setFixedWidth(50)
        go_btn.setStyleSheet('padding: 4px; background: #0078d4; color: white; border: none;')
        go_btn.clicked.connect(self._on_search)

        self.location_combo = QComboBox()
        self.location_combo.setStyleSheet('padding: 4px; background: #2d2d2d; color: white;')
        self.location_combo.addItem('── Select location ──')
        for name in get_location_names(self.locations):
            self.location_combo.addItem(name)
        self.location_combo.currentTextChanged.connect(self._on_location_selected)

        top_bar.addWidget(self.search_box, stretch=3)
        top_bar.addWidget(go_btn)
        top_bar.addWidget(self.location_combo, stretch=2)
        root.addLayout(top_bar)

        # ── Map display ───────────────────────────────────────────────────────
        self.map_label = QLabel('Waiting for map image…')
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

        n = len(self.locations)
        if n:
            self.show_status(f'[INIT] {n} location(s) loaded — ready', 'success')
        else:
            self.show_status('[INIT] No locations loaded — check config/locations.yaml', 'error')

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

    def _on_search(self):
        query = self.search_box.text().strip()
        if not query:
            return
        result = search_location(self.locations, query)
        if result is None:
            self.show_status(f'[SEARCH] No match for "{query}"', 'error')
            return
        name, loc = result
        self.show_status(f'[SEARCH] Found "{name}" — navigating…', 'info')
        self._publish_goal_and_navigate(name, loc)

    def _on_location_selected(self, name: str):
        if name.startswith('──'):
            return
        result = search_location(self.locations, name)
        if result is None:
            self.show_status(f'[DROPDOWN] "{name}" not found in locations', 'error')
            return
        _, loc = result
        self._publish_goal_and_navigate(name, loc)

    def _publish_goal_and_navigate(self, name: str, loc: Dict[str, Any]):
        if self.current_position is None:
            self.show_status('[NAV] No position yet — waiting for current_position', 'error')
            return

        now = self.get_clock().now().to_msg()

        sx = self.current_position.point.x
        sy = self.current_position.point.y
        gx = float(loc['x'])
        gy = float(loc['y'])

        start_msg = PointStamped()
        start_msg.header.stamp    = now
        start_msg.header.frame_id = 'map'
        start_msg.point.x         = sx
        start_msg.point.y         = sy
        self.start_pub.publish(start_msg)

        goal_msg = PointStamped()
        goal_msg.header.stamp    = now
        goal_msg.header.frame_id = 'map'
        goal_msg.point.x         = gx
        goal_msg.point.y         = gy
        self.goal_pub.publish(goal_msg)

        self.show_status(
            f'[NAV] → "{name}"  goal ({gx:.2f}, {gy:.2f})  from ({sx:.2f}, {sy:.2f})',
            'success'
        )

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
