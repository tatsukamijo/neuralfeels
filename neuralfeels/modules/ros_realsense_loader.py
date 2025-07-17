import threading
from typing import Dict, Optional, Tuple
import rospy
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
from cv_bridge import CvBridge
import logging
from neuralfeels.datasets.data_util import FrameData
import torch

logging.basicConfig(level=logging.INFO)

class RealTimeRealsenseDataLoader:
    """
    Subscribes to realsense RGB, depth, and camera_info topics for multiple sensors.
    Keeps the latest data for each sensor and provides methods to retrieve them as FrameData.
    """
    def __init__(self, sensor_configs: Optional[Dict[str, Dict]] = None):
        """
        Args:
            sensor_configs: dict (optional)
                If None, uses default config matching realsense_publisher.py topics
                Example:
                {
                  'realsense_front_left': {
                      'rgb_topic': '/realsense/color/image_raw',
                      'depth_topic': '/realsense/aligned_depth_to_color/image_raw',
                      'camera_info_topic': '/realsense/color/camera_info',
                      'depth_scale': 0.001  # meters per unit
                  },
                  ...
                }
        """
        # Default config matching realsense_publisher.py
        if sensor_configs is None:
            sensor_configs = {
                'realsense_main': {
                    'rgb_topic': '/realsense/color/image_raw',
                    'depth_topic': '/realsense/aligned_depth_to_color/image_raw',
                    'camera_info_topic': '/realsense/color/camera_info',
                    'depth_scale': 0.00011
                }
            }
        
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.sensor_configs = sensor_configs
        self.latest_data: Dict[str, Dict] = {name: {} for name in sensor_configs.keys()}
        self.subscribers = []
        
        # Initialize ROS node only if not already initialized
        try:
            rospy.init_node("realsense_realtime_loader", anonymous=True, disable_signals=False)
        except rospy.exceptions.ROSException:
            # Node already initialized
            pass
            
        for sensor_name, config in sensor_configs.items():
            try:
                rgb_sub = rospy.Subscriber(
                    config['rgb_topic'], Image, self._make_rgb_callback(sensor_name), queue_size=1
                )
                depth_sub = rospy.Subscriber(
                    config['depth_topic'], Image, self._make_depth_callback(sensor_name), queue_size=1
                )
                info_sub = rospy.Subscriber(
                    config['camera_info_topic'], CameraInfo, self._make_info_callback(sensor_name), queue_size=1
                )
                self.subscribers.extend([rgb_sub, depth_sub, info_sub])
                logging.info(f"Subscribed to {sensor_name}: {config['rgb_topic']}, {config['depth_topic']}, {config['camera_info_topic']}")
            except Exception as e:
                logging.error(f"Failed to subscribe to {sensor_name}: {e}")

    def _make_rgb_callback(self, sensor_name: str):
        def callback(msg: Image):
            try:
                # realsense_publisher.py publishes RGB, not BGR
                rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                with self.lock:
                    self.latest_data[sensor_name]['rgb'] = rgb
            except Exception as e:
                logging.error(f"Error processing RGB for {sensor_name}: {e}")
        return callback

    def _make_depth_callback(self, sensor_name: str):
        def callback(msg: Image):
            try:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
                with self.lock:
                    self.latest_data[sensor_name]['depth'] = depth
            except Exception as e:
                logging.error(f"Error processing depth for {sensor_name}: {e}")
        return callback

    def _make_info_callback(self, sensor_name: str):
        def callback(msg: CameraInfo):
            try:
                camera_info = {
                    'width': msg.width,
                    'height': msg.height,
                    'fx': msg.K[0],
                    'fy': msg.K[4],
                    'cx': msg.K[2],
                    'cy': msg.K[5],
                    'distortion_coeffs': list(msg.D)
                }
                with self.lock:
                    self.latest_data[sensor_name]['camera_info'] = camera_info
            except Exception as e:
                logging.error(f"Error processing camera info for {sensor_name}: {e}")
        return callback

    def get_latest_data(self) -> Dict[str, Dict]:
        """Get the latest data for all sensors."""
        with self.lock:
            return {name: data.copy() for name, data in self.latest_data.items()}

    def wait_for_data(self, timeout: float = 10.0) -> bool:
        """Wait for at least one sensor to have both RGB and depth data."""
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                for sensor_name, data in self.latest_data.items():
                    if 'rgb' in data and 'depth' in data and 'camera_info' in data:
                        if data['rgb'] is not None and data['depth'] is not None:
                            logging.info(f"Received data from {sensor_name}")
                            return True
            time.sleep(0.1)
        logging.warning("No realsense data received within timeout.")
        return False

    def get_frame_data(self, idx: int, sensor_name: str, pose: np.ndarray, device: str = "cpu") -> Tuple[FrameData, Dict]:
        """
        Get frame data for a specific sensor in the format expected by RealsenseSensor.
        Returns (FrameData, calibration_dict)
        """
        if sensor_name not in self.sensor_configs:
            raise ValueError(f"Unknown sensor: {sensor_name}")
        
        data = self.latest_data.get(sensor_name, {})
        rgb = data.get('rgb')
        depth = data.get('depth')
        camera_info = data.get('camera_info')
        
        if rgb is None or depth is None or camera_info is None:
            logging.warning(f"No data available for {sensor_name}, using dummy data")
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            depth = np.zeros((480, 640), dtype=np.uint16)
            camera_info = {
                'width': 640,
                'height': 480,
                'fx': 615.0,
                'fy': 615.0,
                'cx': 320.0,
                'cy': 240.0,
                'distortion_coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]
            }
        
        # Convert depth to float32 using scale
        depth_scale = self.sensor_configs[sensor_name]['depth_scale']
        depth_float = depth.astype(np.float32) * depth_scale
        
        # RGB is already in RGB format from realsense_publisher.py
        rgb_rgb = rgb.copy()  # Already RGB, no need to convert
        
        # Prepare data in the format expected by RealsenseSensor
        im_np = rgb_rgb[None, ...]  # (1, H, W, C)
        depth_np = depth_float[None, ...]  # (1, H, W)
        im = torch.from_numpy(im_np).float().to(device) / 255.0
        depth_tensor = torch.from_numpy(depth_np).float().to(device)
        T_np = pose[None, ...]  # (1, 4, 4)
        T = torch.from_numpy(T_np).float().to(device)
        
        calibration = {
            'intrinsics': camera_info,
            'pose': T,
            'depth_scale': depth_scale
        }
        
        frame_data = FrameData(
            frame_id=np.array([idx]),
            im_batch=im,
            im_batch_np=im_np,
            depth_batch=depth_tensor,
            depth_batch_np=depth_np,
            T_WC_batch=T,
            T_WC_batch_np=T_np,
            seg_pixels=None,
            format=[sensor_name],
            frame_avg_losses=torch.zeros(1, device=device),
        )
        return frame_data, calibration

    def get_available_sensors(self) -> list:
        """Get list of sensors that have received data."""
        with self.lock:
            available = []
            for sensor_name, data in self.latest_data.items():
                if 'rgb' in data and 'depth' in data and 'camera_info' in data:
                    if data['rgb'] is not None and data['depth'] is not None:
                        available.append(sensor_name)
            return available 