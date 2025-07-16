from neuralfeels.modules.ros_digit_loader import RealTimeDigitDataLoader
import time

def main():
    loader = RealTimeDigitDataLoader()
    print("Waiting for any DIGIT image...")
    if loader.wait_for_any_image(timeout=5.0):
        print("At least one DIGIT image received!")
    else:
        print("No DIGIT images received within timeout.")

    for i in range(10):
        images = loader.get_latest_images()
        for idx, img in images.items():
            if img is not None:
                print(f"Digit {idx}: image shape = {img.shape}")
            else:
                print(f"Digit {idx}: no image")
        time.sleep(1)

if __name__ == "__main__":
    main() 