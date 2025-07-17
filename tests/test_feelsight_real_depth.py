#!/usr/bin/env python3
"""
Test script to visualize TactileDepth model performance on feelsight_real dataset.
This script loads real DIGIT data from the feelsight_real dataset and compares
the TactileDepth model output with ground truth depth.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import torch
import pickle
from pathlib import Path
import argparse
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

# Add neuralfeels to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from neuralfeels.contrib.tactile_transformer.tactile_depth import TactileDepth
from neuralfeels.datasets import image_transforms
from neuralfeels.datasets.data_util import FrameData


def load_feelsight_real_data(dataset_path, object_name, log_id, max_frames=10):
    """
    Load DIGIT data from feelsight_real dataset.
    
    Args:
        dataset_path (str): Path to feelsight_real dataset
        object_name (str): Object name (e.g., 'bell_pepper')
        log_id (str): Log ID (e.g., '00')
        max_frames (int): Maximum number of frames to load
    
    Returns:
        dict: Dictionary containing DIGIT data for each sensor
    """
    print(f"Loading feelsight_real data: {object_name}/{log_id}")
    
    # Load dataset info
    pkl_path = os.path.join(dataset_path, "data.pkl")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    digit_info = data["digit_info"]
    print(f"Dataset info loaded: {len(data['time'])} frames")
    
    # Load DIGIT data for each sensor
    digit_data = {}
    sensor_names = ["digit_thumb", "digit_index", "digit_middle", "digit_ring"]
    sensor_locations = ["thumb", "index", "middle", "ring"]
    
    for sensor_name, sensor_location in zip(sensor_names, sensor_locations):
        seq_dir = os.path.join(dataset_path, "allegro", sensor_location, "image")
        
        if not os.path.exists(seq_dir):
            print(f"Warning: {seq_dir} does not exist, skipping {sensor_name}")
            continue
        
        # Load RGB image files (feelsight_real uses .jpg format)
        rgb_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
        
        if len(rgb_files) == 0:
            print(f"Warning: No image files found in {seq_dir}")
            continue
        
        # Limit frames
        n_frames = min(len(rgb_files), max_frames)
        rgb_files = rgb_files[:n_frames]
        
        digit_data[sensor_name] = {
            'rgb_files': rgb_files,
            'seq_dir': seq_dir,
            'digit_info': digit_info
        }
        
        print(f"  {sensor_name}: {n_frames} frames")
    
    return digit_data


def load_frame_data(digit_data, sensor_name, frame_idx):
    """
    Load a single frame of DIGIT data.
    
    Args:
        digit_data (dict): DIGIT data dictionary
        sensor_name (str): Sensor name
        frame_idx (int): Frame index
    
    Returns:
        tuple: (rgb_image, gt_depth, pose)
    """
    if sensor_name not in digit_data:
        return None, None, None
    
    data = digit_data[sensor_name]
    rgb_file = data['rgb_files'][frame_idx]
    seq_dir = data['seq_dir']
    
    # Load RGB image
    rgb_path = os.path.join(seq_dir, rgb_file)
    rgb_image = cv2.imread(rgb_path)
    if rgb_image is None:
        print(f"Warning: Could not load {rgb_path}")
        return None, None, None
    
    # For feelsight_real, we don't have ground truth depth, so return None
    gt_depth = None
    
    # Load pose (dummy for now - would need to load from allegro data)
    pose = np.eye(4, dtype=np.float32)
    
    return rgb_image, gt_depth, pose


def apply_depth_transforms(depth, digit_info):
    """
    Apply the same depth transforms as used in the real-time loader.
    
    Args:
        depth (np.ndarray): Raw depth from TactileDepth model
        digit_info (dict): DIGIT calibration info
    
    Returns:
        np.ndarray: Transformed depth
    """
    # Apply same transforms as in ros_digit_loader
    inv_depth_scale = 1.0 / digit_info["depth_scale"]
    cam_dist = digit_info["cam_dist"]
    
    # Create transforms
    depth_scale = image_transforms.DepthScale(inv_depth_scale)
    depth_transform = image_transforms.DepthTransform(cam_dist)
    
    # Apply transforms
    depth_scaled = depth_scale(depth)
    depth_transformed = depth_transform(depth_scaled)
    
    return depth_transformed


def visualize_comparison(rgb_image, gt_depth, pred_depth, sensor_name, frame_idx, save_path=None):
    """
    Visualize RGB image, ground truth depth, and predicted depth side by side.
    
    Args:
        rgb_image (np.ndarray): RGB image
        gt_depth (np.ndarray): Ground truth depth (can be None for feelsight_real)
        pred_depth (np.ndarray): Predicted depth from TactileDepth
        sensor_name (str): Sensor name
        frame_idx (int): Frame index
        save_path (str): Path to save visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{sensor_name} - Frame {frame_idx}', fontsize=16)
    
    # RGB image
    axes[0, 0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')
    
    # Ground truth depth
    if gt_depth is not None:
        gt_valid = gt_depth[~np.isnan(gt_depth)]
        if len(gt_valid) > 0:
            gt_min, gt_max = gt_valid.min(), gt_valid.max()
            gt_normalized = (gt_depth - gt_min) / (gt_max - gt_min + 1e-8)
            gt_normalized = np.clip(gt_normalized, 0, 1)
            gt_colored = cm.viridis(gt_normalized)
            gt_colored[np.isnan(gt_depth), :] = [1, 1, 1, 1]  # White for NaN
            axes[0, 1].imshow(gt_colored)
            axes[0, 1].set_title(f'Ground Truth Depth\nRange: [{gt_min:.4f}, {gt_max:.4f}]')
        else:
            axes[0, 1].text(0.5, 0.5, 'No GT depth data', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Ground Truth Depth (No Data)')
    else:
        axes[0, 1].text(0.5, 0.5, 'GT depth not available\n(feelsight_real dataset)', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Ground Truth Depth (Not Available)')
    axes[0, 1].axis('off')
    
    # Predicted depth
    if pred_depth is not None:
        pred_valid = pred_depth[~np.isnan(pred_depth)]
        if len(pred_valid) > 0:
            pred_min, pred_max = pred_valid.min(), pred_valid.max()
            pred_normalized = (pred_depth - pred_min) / (pred_max - pred_min + 1e-8)
            pred_normalized = np.clip(pred_normalized, 0, 1)
            pred_colored = cm.viridis(pred_normalized)
            pred_colored[np.isnan(pred_depth), :] = [1, 1, 1, 1]  # White for NaN
            axes[0, 2].imshow(pred_colored)
            axes[0, 2].set_title(f'Predicted Depth\nRange: [{pred_min:.4f}, {pred_max:.4f}]')
        else:
            axes[0, 2].text(0.5, 0.5, 'No predicted depth data', ha='center', va='center', transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Predicted Depth (No Data)')
    else:
        axes[0, 2].text(0.5, 0.5, 'Prediction failed', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('Predicted Depth (Failed)')
    axes[0, 2].axis('off')
    
    # Histograms
    if gt_depth is not None and len(gt_valid) > 0:
        axes[1, 0].hist(gt_valid.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 0].set_title('Ground Truth Depth Histogram')
        axes[1, 0].set_xlabel('Depth (meters)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No GT depth data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Ground Truth Depth Histogram')
    
    if pred_depth is not None and len(pred_valid) > 0:
        axes[1, 1].hist(pred_valid.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1, 1].set_title('Predicted Depth Histogram')
        axes[1, 1].set_xlabel('Depth (meters)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No predicted depth data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Predicted Depth Histogram')
    
    # Statistics comparison
    if gt_depth is not None and pred_depth is not None and len(gt_valid) > 0 and len(pred_valid) > 0:
        stats_text = f'Ground Truth:\n'
        stats_text += f'  Mean: {gt_valid.mean():.4f}m\n'
        stats_text += f'  Std: {gt_valid.std():.4f}m\n'
        stats_text += f'  Min: {gt_valid.min():.4f}m\n'
        stats_text += f'  Max: {gt_valid.max():.4f}m\n'
        stats_text += f'  Pixels: {len(gt_valid)}\n\n'
        stats_text += f'Predicted:\n'
        stats_text += f'  Mean: {pred_valid.mean():.4f}m\n'
        stats_text += f'  Std: {pred_valid.std():.4f}m\n'
        stats_text += f'  Min: {pred_valid.min():.4f}m\n'
        stats_text += f'  Max: {pred_valid.max():.4f}m\n'
        stats_text += f'  Pixels: {len(pred_valid)}'
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        axes[1, 2].set_title('Statistics Comparison')
        axes[1, 2].axis('off')
    elif pred_depth is not None and len(pred_valid) > 0:
        # Only show predicted statistics when GT is not available
        stats_text = f'Predicted Depth:\n'
        stats_text += f'  Mean: {pred_valid.mean():.4f}m\n'
        stats_text += f'  Std: {pred_valid.std():.4f}m\n'
        stats_text += f'  Min: {pred_valid.min():.4f}m\n'
        stats_text += f'  Max: {pred_valid.max():.4f}m\n'
        stats_text += f'  Pixels: {len(pred_valid)}\n\n'
        stats_text += f'Note: No ground truth\n'
        stats_text += f'available for feelsight_real'
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 2].set_title('Predicted Statistics')
        axes[1, 2].axis('off')
    else:
        axes[1, 2].text(0.5, 0.5, 'No data for comparison', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Statistics Comparison')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def test_feelsight_real_depth(dataset_path, object_name, log_id, max_frames=5, save_dir=None):
    """
    Test TactileDepth model on feelsight_real dataset.
    
    Note: feelsight_real dataset contains only RGB images (.jpg files) and no ground truth depth.
    This script will generate depth predictions using TactileDepth model and visualize the results.
    
    Args:
        dataset_path (str): Path to feelsight_real dataset
        object_name (str): Object name
        log_id (str): Log ID
        max_frames (int): Maximum number of frames to test
        save_dir (str): Directory to save visualizations
    """
    print(f"Testing TactileDepth on feelsight_real: {object_name}/{log_id}")
    print("Note: feelsight_real dataset contains only RGB images, no ground truth depth available")
    
    # Load dataset data
    digit_data = load_feelsight_real_data(dataset_path, object_name, log_id, max_frames)
    
    if not digit_data:
        print("No DIGIT data found!")
        return
    
    # Initialize Hydra for TactileDepth model
    print("Initializing Hydra for TactileDepth model...")
    hydra.initialize(config_path="../scripts/config", version_base=None)
    
    # Initialize TactileDepth model
    print("Initializing TactileDepth model...")
    tac_depth = TactileDepth(depth_mode="vit", real=True, device="cpu")
    print("✓ TactileDepth model initialized")
    
    # Create save directory
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each sensor
    for sensor_name in digit_data.keys():
        print(f"\n=== Testing {sensor_name} ===")
        data = digit_data[sensor_name]
        n_frames = len(data['rgb_files'])
        
        for frame_idx in tqdm(range(n_frames), desc=f"Processing {sensor_name}"):
            # Load frame data
            rgb_image, gt_depth, pose = load_frame_data(digit_data, sensor_name, frame_idx)
            
            if rgb_image is None:
                print(f"  Frame {frame_idx}: Failed to load data")
                continue
            
            print(f"  Frame {frame_idx}: RGB shape={rgb_image.shape}")
            
            try:
                # Generate depth prediction
                img_bgr = rgb_image[:, :, ::-1]  # RGB -> BGR
                pred_depth_raw = tac_depth.image2heightmap(img_bgr, sensor_name=sensor_name)
                
                # Generate mask
                mask = tac_depth.heightmap2mask(pred_depth_raw, sensor_name=sensor_name)
                
                # Apply mask
                pred_depth_np = pred_depth_raw.cpu().numpy().astype(np.float32)
                mask_np = mask.cpu().numpy().astype(np.float32)
                pred_depth_masked = pred_depth_np * mask_np
                
                print(f"    Raw depth range: [{pred_depth_np.min():.4f}, {pred_depth_np.max():.4f}]")
                print(f"    Masked depth range: [{pred_depth_masked.min():.4f}, {pred_depth_masked.max():.4f}]")
                print(f"    Non-zero pixels: {np.sum(pred_depth_masked != 0)}")
                
                # Apply depth transforms (same as real-time loader)
                pred_depth_transformed = apply_depth_transforms(pred_depth_masked, data['digit_info'])
                
                print(f"    Transformed depth range: [{pred_depth_transformed.min():.4f}, {pred_depth_transformed.max():.4f}]")
                print(f"    Valid pixels: {np.sum(~np.isnan(pred_depth_transformed))}")
                
                # Debug: Show detailed depth values
                if np.sum(pred_depth_masked != 0) > 0:
                    non_zero_depths = pred_depth_masked[pred_depth_masked != 0]
                    print(f"    Non-zero depth values: min={non_zero_depths.min():.6}, max={non_zero_depths.max():.6f}")
                    print(f"    Non-zero depth values after scale: min={(non_zero_depths * data['digit_info']['depth_scale']).min():.6f}, max={(non_zero_depths * data['digit_info']['depth_scale']).max():.6f}")
                    print(f"    Non-zero depth values after transform: min={(non_zero_depths * data['digit_info']['depth_scale'] + data['digit_info']['cam_dist']).min():.6f}, max={(non_zero_depths * data['digit_info']['depth_scale'] + data['digit_info']['cam_dist']).max():.6f}")
                    print(f"    cam_dist: {data['digit_info']['cam_dist']}")
                    print(f"    depth_scale: {data['digit_info']['depth_scale']}")
                
                # Visualize
                if save_dir:
                    save_path = save_dir / f"{sensor_name}_frame{frame_idx:03d}.png"
                else:
                    save_path = None
                
                visualize_comparison(
                    rgb_image, gt_depth, pred_depth_transformed,
                    sensor_name, frame_idx, save_path
                )
                
            except Exception as e:
                print(f"    Error processing frame {frame_idx}: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n=== Test completed ===")
    print("Note: This test shows TactileDepth predictions on real DIGIT data.")
    print("No ground truth depth is available for comparison in feelsight_real dataset.")


def main():
    parser = argparse.ArgumentParser(description="Test TactileDepth on feelsight_real dataset")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to feelsight_real dataset")
    parser.add_argument("--object", type=str, default="bell_pepper",
                       help="Object name (default: bell_pepper)")
    parser.add_argument("--log_id", type=str, default="00",
                       help="Log ID (default: 00)")
    parser.add_argument("--max_frames", type=int, default=3,
                       help="Maximum number of frames to test (default: 3)")
    parser.add_argument("--save_dir", type=str, default=None,
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path {args.dataset_path} does not exist!")
        return
    
    # Run test
    test_feelsight_real_depth(
        dataset_path=args.dataset_path,
        object_name=args.object,
        log_id=args.log_id,
        max_frames=args.max_frames,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main() 