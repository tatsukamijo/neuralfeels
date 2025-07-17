import threading
from typing import Dict, Optional
import rospy
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
import logging
from neuralfeels.datasets.data_util import FrameData
import torch

logging.basicConfig(level=logging.INFO)

class RealTimeDigitDataLoader:
    """
    Subscribes to /digit/0/image_raw through /digit/3/image_raw and keeps the latest image for each.
    Provides a method to get the latest images at any time.
    Warns if a topic is missing or no data is received.
    """
    DIGIT_TOPICS = [f"/digit/{i}/image_raw" for i in range(4)]
    DIGIT_NAMES = ["digit_thumb", "digit_index", "digit_middle", "digit_ring"]
    IMG_HEIGHT = 240
    IMG_WIDTH = 320
    N_CHANNELS = 3
    
    # Offline data parameters from feelsight_real/bell_pepper/00/data.pkl
    DEPTH_SCALE = 33333.33334
    CAM_DIST = -0.022

    def __init__(self) -> None:
        """Initializes ROS subscribers and image buffers."""
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_images: Dict[int, Optional[np.ndarray]] = {i: None for i in range(4)}
        self.subscribers = []
        rospy.init_node("digit_realtime_loader", anonymous=True, disable_signals=False)
        for i, topic in enumerate(self.DIGIT_TOPICS):
            try:
                sub = rospy.Subscriber(topic, Image, self._make_callback(i), queue_size=1)
                self.subscribers.append(sub)
            except Exception as e:
                print(f"Failed to subscribe to {topic}: {e}")

    def _make_callback(self, idx: int):
        def callback(msg: Image):
            try:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                # print(f"[ros_digit_loader] imgmsg_to_cv2 output shape: {img.shape} for digit_{idx}")
                # print(f"[ros_digit_loader] img dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
                with self.lock:
                    self.latest_images[idx] = img
            except Exception as e:
                print(f"Error converting image for digit {idx}: {e}")
        return callback

    def get_latest_images(self) -> Dict[int, Optional[np.ndarray]]:
        """
        Returns the latest images for each digit sensor.
        Returns:
            Dict[int, Optional[np.ndarray]]: Mapping from digit index to latest image (or None if not received).
        """
        with self.lock:
            return dict(self.latest_images)

    def wait_for_any_image(self, timeout: float = 5.0) -> bool:
        """
        Waits until at least one image is received or timeout.
        Returns:
            bool: True if at least one image was received, False if timeout.
        """
        import time
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                if any(img is not None for img in self.latest_images.values()):
                    return True
            time.sleep(0.1)
        print("No DIGIT images received within timeout.")
        return False

    def get_frame_data(self, idx: int, pose: np.ndarray, digit_idx: int = 0, device: str = "cpu") -> FrameData:
        digit_name = self.DIGIT_NAMES[digit_idx]
        img = self.get_latest_images().get(digit_idx)
        if img is None:
            img = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, self.N_CHANNELS), dtype=np.uint8)
        # Only print if transpose or error
        # Handle image shape - DIGIT images should be 240x320 (HxW)
        if img.shape == (self.IMG_WIDTH, self.IMG_HEIGHT, self.N_CHANNELS):
            img = np.transpose(img, (1, 0, 2))
            print(f"[ros_digit_loader] img transposed from {self.IMG_WIDTH}x{self.IMG_HEIGHT} to {img.shape} for {digit_name} at idx={idx}")
        elif img.shape != (self.IMG_HEIGHT, self.IMG_WIDTH, self.N_CHANNELS):
            print(f"[ros_digit_loader] Unexpected img.shape={img.shape} for {digit_name} at idx={idx}")
            print(f"[ros_digit_loader] Expected shape: ({self.IMG_HEIGHT}, {self.IMG_WIDTH}, {self.N_CHANNELS})")
            raise ValueError(f"[ros_digit_loader] Unexpected img.shape={img.shape}")
        print(f"[ros_digit_loader] get_frame_data: idx={idx}, digit={digit_name}, img_shape={img.shape}, t={__import__('time').time()}")
        
        # Generate depth information from RGB image using offline DigitSensor approach
        try:
            # Create a minimal DigitSensor configuration for real-time processing
            from omegaconf import DictConfig
            import copy
            
            # Create digit_info similar to offline data
            digit_info = {
                'depth_scale': self.DEPTH_SCALE,
                'cam_dist': self.CAM_DIST,
                'intrinsics': {
                    'w': self.IMG_WIDTH,
                    'h': self.IMG_HEIGHT,
                    'fx': 277.1281292110204,  # From offline data
                    'fy': 277.1281292110204,
                    'cx': 120.0,
                    'cy': 160.0
                }
            }
            
            # Create sensor config similar to offline (complete config from digit.yaml)
            sensor_config = DictConfig({
                'name': digit_name,
                'tactile_depth': {
                    'mode': 'vit',
                    'use_real_data': True
                },
                'sampling': {
                    'n_rays': 5,
                    'n_strat_samples': 10,
                    'n_surf_samples': 10,
                    'depth_range': [-0.01, 0.05],
                    'surface_samples_offset': 1e-3,
                    'dist_behind_surf': 2e-2,
                    'loss_ratio': 0.1,
                    'free_space_ratio': 0.0
                },
                'kf_min_loss': 1e-2,
                'gel': {
                    'origin': [0.022, 0, 0],
                    'width': 0.02,
                    'height': 0.03,
                    'curvature': True,
                    'curvatureMax': 0.004,
                    'R': 0.1,
                    'countW': 100
                },
                'viz': {
                    'reduce_factor': 1,
                    'reduce_factor_up': 1
                }
            })
            
            # Import and create DigitSensor (same as offline)
            from neuralfeels.modules.sensor import DigitSensor
            
            # Create temporary DigitSensor with ROS data
            temp_sensor = DigitSensor(
                cfg_sensor=sensor_config,
                dataset_path=None,  # No dataset path for real-time
                calibration=digit_info,  # Use our digit_info
                device=device
            )
            
            # Force background template regeneration for better contact detection
            if hasattr(temp_sensor.tac_depth, 'background_images'):
                if digit_name in temp_sensor.tac_depth.background_images:
                    print(f"[DEBUG] Removing existing background template for {digit_name}")
                    del temp_sensor.tac_depth.background_images[digit_name]
            
            # Create msg_data format like offline mode
            msg_data = {"color": img}
            
            # Debug: Check depth before transform
            print(f"[DEBUG] Using offline DigitSensor approach")
            print(f"[DEBUG] temp_sensor.depth_transform: {temp_sensor.depth_transform}")
            print(f"[DEBUG] temp_sensor.inv_depth_scale: {temp_sensor.inv_depth_scale}")
            print(f"[DEBUG] temp_sensor.cam_dist: {temp_sensor.cam_dist}")
            
            # Use the same get_frame_data method as offline DigitSensor
            frame_data = temp_sensor.get_frame_data(
                idx=idx,
                poses=pose,
                msg_data=msg_data
            )
            
            # Extract depth from the frame data
            depth = frame_data.depth_batch_np[0]  # (H, W)
            
            print(f"[DEBUG] Final depth: shape={depth.shape}, min={depth.min():.6f}, max={depth.max():.6f}")
            print(f"[DEBUG] NaN count: {np.isnan(depth).sum()}, non-NaN count: {np.count_nonzero(~np.isnan(depth))}")
            
        except Exception as e:
            print(f"[ros_digit_loader] Failed to generate depth using offline approach: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to NaN depth if generation fails
            depth = np.full((self.IMG_HEIGHT, self.IMG_WIDTH), np.nan, dtype=np.float32)
        
        # Ensure depth has correct shape (240x320)
        if depth.shape != (self.IMG_HEIGHT, self.IMG_WIDTH):
            print(f"[ros_digit_loader] depth.shape={depth.shape} is not ({self.IMG_HEIGHT},{self.IMG_WIDTH}), transposing if possible.")
            if depth.shape == (self.IMG_WIDTH, self.IMG_HEIGHT):
                depth = depth.T
                print(f"[ros_digit_loader] depth transposed to {depth.shape}")
            else:
                raise ValueError(f"[ros_digit_loader] Unexpected depth.shape={depth.shape}")
        
        im_np = img[None, ...]  # (1, H, W, C)
        depth_np = depth[None, ...]  # (1, H, W)
        T_np = pose[None, ...]  # (1, 4, 4)
        im = torch.from_numpy(im_np).float().to(device) / 255.0
        depth_t = torch.from_numpy(depth_np).float().to(device)
        T = torch.from_numpy(T_np).float().to(device)
        data = FrameData(
            frame_id=np.array([idx]),
            im_batch=im,
            im_batch_np=im_np,
            depth_batch=depth_t,
            depth_batch_np=depth_np,
            T_WC_batch=T,
            T_WC_batch_np=T_np,
            format=[digit_name],
            frame_avg_losses=torch.zeros([1], device=device),
        )
        return data 