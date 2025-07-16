import threading
from typing import Dict, Optional
import rospy
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge
import logging

logging.basicConfig(level=logging.INFO)

class RealTimeDigitDataLoader:
    """
    Subscribes to /digit/0/image_raw through /digit/3/image_raw and keeps the latest image for each.
    Provides a method to get the latest images at any time.
    Warns if a topic is missing or no data is received.
    """
    DIGIT_TOPICS = [f"/digit/{i}/image_raw" for i in range(4)]

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
                logging.warning(f"Failed to subscribe to {topic}: {e}")

    def _make_callback(self, idx: int):
        def callback(msg: Image):
            try:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                with self.lock:
                    self.latest_images[idx] = img
            except Exception as e:
                logging.warning(f"Error converting image for digit {idx}: {e}")
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
        logging.warning("No DIGIT images received within timeout.")
        return False 