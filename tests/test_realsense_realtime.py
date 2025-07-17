from neuralfeels.modules.ros_realsense_loader import RealTimeRealsenseDataLoader
import numpy as np

loader = RealTimeRealsenseDataLoader()
loader.wait_for_data(timeout=10)
frame_data, calib = loader.get_frame_data(0, "realsense_main", np.eye(4), device="cpu")
print("RGB shape:", frame_data.im_batch_np.shape)
print("Depth shape:", frame_data.depth_batch_np.shape)
print("Calibration:", calib)
