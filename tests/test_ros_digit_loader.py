from neuralfeels.modules.ros_digit_loader import RealTimeDigitDataLoader
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import hydra
from omegaconf import DictConfig

def test_ros_connection():
    """Test basic ROS connection and topic availability"""
    print("=== Testing ROS Connection ===")
    
    try:
        import rospy
        print("✓ rospy imported successfully")
        
        # Try to get ROS master info
        try:
            master = rospy.get_master()
            print(f"✓ ROS master found: {master.getUri()}")
        except Exception as e:
            print(f"✗ ROS master connection failed: {e}")
            return False
            
        # Try to get topic list
        try:
            topics = rospy.get_published_topics()
            print(f"✓ Found {len(topics)} published topics")
            
            # Look for DIGIT topics
            digit_topics = [topic for topic, msg_type in topics if 'digit' in topic.lower()]
            if digit_topics:
                print(f"✓ Found DIGIT topics: {digit_topics}")
            else:
                print("✗ No DIGIT topics found")
                print("Available topics:")
                for topic, msg_type in topics[:10]:  # Show first 10 topics
                    print(f"  - {topic} ({msg_type})")
                if len(topics) > 10:
                    print(f"  ... and {len(topics) - 10} more topics")
                    
        except Exception as e:
            print(f"✗ Failed to get topic list: {e}")
            return False
            
        return True
        
    except ImportError as e:
        print(f"✗ rospy import failed: {e}")
        return False

def test_dummy_depth_generation():
    """Test depth generation with dummy images (no ROS required)"""
    print("\n=== Testing Depth Generation with Dummy Images ===")
    
    try:
        # Initialize Hydra properly
        hydra.initialize(config_path="../scripts/config", version_base=None)
        
        from neuralfeels.contrib.tactile_transformer.tactile_depth import TactileDepth
        
        # Create dummy RGB image (simulating DIGIT image)
        dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print(f"Created dummy image: shape={dummy_img.shape}")
        
        # Initialize TactileDepth model
        tac_depth = TactileDepth(depth_mode="vit", real=True, device="cpu")
        print("✓ TactileDepth model initialized")
        
        # Test depth generation
        img_bgr = dummy_img[:, :, ::-1]  # RGB -> BGR
        depth = tac_depth.image2heightmap(img_bgr, sensor_name="test_digit")
        print(f"✓ Depth generated: shape={depth.shape}, type={type(depth)}")
        
        # Test mask generation
        mask = tac_depth.heightmap2mask(depth, sensor_name="test_digit")
        print(f"✓ Mask generated: shape={mask.shape}, type={type(mask)}")
        
        # Apply mask
        depth_np = depth.cpu().numpy().astype(np.float32)
        mask_np = mask.cpu().numpy().astype(np.float32)
        depth_masked = depth_np * mask_np
        
        print(f"✓ Final depth: shape={depth_masked.shape}, range=[{depth_masked.min():.4f}, {depth_masked.max():.4f}]")
        
        # Visualize dummy results
        visualize_depth_and_image(dummy_img, depth_masked, "dummy_digit", "test_dummy_depth_visualization.png")
        
        return True
        
    except Exception as e:
        print(f"✗ Dummy depth generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_digit_image_reception():
    """Test DIGIT image reception with detailed debugging"""
    print("\n=== Testing DIGIT Image Reception ===")
    
    try:
        loader = RealTimeDigitDataLoader()
        print("✓ RealTimeDigitDataLoader initialized")
        
        # Test immediate image reception
        print("Testing immediate image reception...")
        images = loader.get_latest_images()
        print(f"Immediate images: {len(images)} received")
        for idx, img in images.items():
            if img is not None:
                print(f"  Digit {idx}: shape={img.shape}, dtype={img.dtype}")
            else:
                print(f"  Digit {idx}: None")
        
        # Test waiting for images
        print("\nWaiting for DIGIT images (timeout: 10s)...")
        start_time = time.time()
        
        while time.time() - start_time < 10.0:
            images = loader.get_latest_images()
            received_count = sum(1 for img in images.values() if img is not None)
            
            if received_count > 0:
                print(f"✓ Received {received_count} DIGIT images!")
                for idx, img in images.items():
                    if img is not None:
                        print(f"  Digit {idx}: shape={img.shape}, dtype={img.dtype}")
                        # Show some pixel values for debugging
                        print(f"    Pixel range: [{img.min()}, {img.max()}]")
                        print(f"    Sample pixels: {img[0, 0, :]}")
                return True
            
            print(f"Waiting... ({time.time() - start_time:.1f}s elapsed)")
            time.sleep(0.5)
        
        print("✗ No DIGIT images received within 10 seconds")
        return False
        
    except Exception as e:
        print(f"✗ DIGIT image reception test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_depth_and_image(rgb_img, depth_img, digit_name, save_path=None):
    """Visualize RGB image and generated depth side by side"""
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB image
    ax1.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'{digit_name} - RGB Image')
    ax1.axis('off')
    
    # Depth visualization with colormap
    # For DIGIT sensors, background pixels are NaN, contact pixels have negative values
    valid_depths = depth_img[~np.isnan(depth_img)]  # Only exclude NaN (background)
    if len(valid_depths) > 0:
        # Use viridis colormap for better depth visualization
        depth_normalized = depth_img.copy()
        depth_normalized[np.isnan(depth_normalized)] = 0  # Replace NaN with 0 for normalization
        
        # Improve color variation by using a more focused range
        if len(valid_depths) > 0:
            # Use actual depth range instead of 0-1 normalization for better color variation
            min_valid = np.min(valid_depths)
            max_valid = np.max(valid_depths)
            depth_normalized = (depth_normalized - min_valid) / (max_valid - min_valid + 1e-8)
            depth_normalized = np.clip(depth_normalized, 0, 1)  # Ensure range [0, 1]
        
        depth_colored = cm.viridis(depth_normalized)
        depth_colored[np.isnan(depth_img), :] = [1, 1, 1, 1]  # White for NaN areas
        im = ax2.imshow(depth_colored)
        ax2.set_title(f'{digit_name} - Depth (Colored)')
        ax2.axis('off')
        
        # Add border around depth plot
        ax2.add_patch(plt.Rectangle((0, 0), depth_img.shape[1], depth_img.shape[0], 
                                   fill=False, edgecolor='black', linewidth=2))
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Depth (normalized)', rotation=270, labelpad=15)
        
        # Depth histogram with better description
        ax3.hist(valid_depths.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_title(f'{digit_name} - Depth Histogram')
        ax3.set_xlabel('Depth Value (meters)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_depth = np.mean(valid_depths)
        max_depth = np.max(valid_depths)
        min_depth = np.min(valid_depths)
        stats_text = f'Mean: {mean_depth:.4f}m\nMax: {max_depth:.4f}m\nMin: {min_depth:.4f}m\nPixels: {len(valid_depths)}'
        ax3.text(0.02, 0.98, stats_text, transform=ax3.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No depth data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'{digit_name} - Depth (No Data)')
        ax2.axis('off')
        
        ax3.text(0.5, 0.5, 'No depth data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title(f'{digit_name} - Depth Histogram')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()

def test_depth_generation():
    """Test depth generation from DIGIT images"""
    print("=== Testing Real DIGIT Images ===")
    
    loader = RealTimeDigitDataLoader()
    
    print("Waiting for any DIGIT image...")
    if loader.wait_for_any_image(timeout=5.0):
        print("✓ At least one DIGIT image received!")
    else:
        print("✗ No DIGIT images received within timeout.")
        return False

    # Test depth generation for each frame
    for frame_idx in range(3):  # Test 3 frames
        print(f"\n--- Frame {frame_idx + 1} ---")
        
        images = loader.get_latest_images()
        if not images:
            print("No images received")
            continue
            
        for digit_idx, img in images.items():
            if img is None:
                print(f"Digit {digit_idx}: no image")
                continue
                
            print(f"Digit {digit_idx}: image shape = {img.shape}")
            
            # Generate frame data with depth
            try:
                # Create dummy pose for testing
                dummy_pose = np.eye(4, dtype=np.float32)
                
                # Get frame data (this will generate depth)
                frame_data = loader.get_frame_data(
                    idx=frame_idx, 
                    pose=dummy_pose, 
                    digit_idx=digit_idx, 
                    device="cpu"
                )
                
                # Extract RGB and depth data
                rgb_img = frame_data.im_batch_np[0]  # (H, W, C)
                depth_img = frame_data.depth_batch_np[0]  # (H, W)
                
                # Convert RGB from normalized to uint8
                rgb_img = (rgb_img * 255).astype(np.uint8)
                
                print(f"  RGB shape: {rgb_img.shape}, range: [{rgb_img.min()}, {rgb_img.max()}]")
                print(f"  Depth shape: {depth_img.shape}")
                
                if not np.all(np.isnan(depth_img)):
                    # For DIGIT sensors, background pixels are NaN, contact pixels have negative values
                    valid_depths = depth_img[~np.isnan(depth_img)]  # Only exclude NaN (background)
                    if len(valid_depths) > 0:
                        print(f"  Depth range: [{depth_img.min():.4f}, {depth_img.max():.4f}]")
                        print(f"  Valid depth pixels: {len(valid_depths)}/{depth_img.size}")
                        print(f"  Contact depth range: [{valid_depths.min():.4f}, {valid_depths.max():.4f}]")
                    else:
                        print(f"  No valid depth data (all background or NaN)")
                else:
                    print(f"  All depth values are NaN")
                
                # Visualize
                digit_name = f"digit_{digit_idx}"
                save_path = f"test_depth_visualization_frame{frame_idx}_{digit_name}.png"
                visualize_depth_and_image(rgb_img, depth_img, digit_name, save_path)
                
            except Exception as e:
                print(f"  Error generating depth for digit {digit_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        time.sleep(1)  # Wait before next frame
    
    return True

def main():
    """Main test function"""
    print("Testing RealTimeDigitDataLoader with depth generation...")
    
    # Test ROS connection first
    ros_ok = test_ros_connection()
    
    # Test DIGIT image reception
    digit_ok = test_digit_image_reception()
    
    # Test dummy depth generation (always works)
    dummy_ok = test_dummy_depth_generation()
    
    # Test real DIGIT images (only if ROS and DIGIT are working)
    real_ok = False
    if ros_ok and digit_ok:
        real_ok = test_depth_generation()
    else:
        print("\n⚠️  Skipping real DIGIT test due to ROS/DIGIT issues")
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"ROS Connection: {'✓' if ros_ok else '✗'}")
    print(f"DIGIT Image Reception: {'✓' if digit_ok else '✗'}")
    print(f"Dummy Depth Generation: {'✓' if dummy_ok else '✗'}")
    print(f"Real DIGIT Depth Generation: {'✓' if real_ok else '✗'}")
    
    if dummy_ok:
        print("\n✅ TactileDepth model is working correctly!")
    else:
        print("\n❌ TactileDepth model has issues")
    
    if digit_ok:
        print("✅ DIGIT image reception is working!")
    else:
        print("❌ DIGIT image reception has issues")
    
    print("Test completed!")

if __name__ == "__main__":
    main() 