import cv2
import time
for camera_index in range(10):
    print(f"\nTesting camera at index {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Could not open camera at index {camera_index}")
        cap.release()
        continue
    
    print(f"Camera {camera_index} opened successfully!")
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print(f"Could not read frame from camera {camera_index}")
        cap.release()
        continue

    window_name = f"Camera {camera_index} Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, frame)
    print(f"Frame displayed from camera {camera_index}. Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    cap.release()
    
    print(f"Camera {camera_index} test completed.") 
print("\nAll camera tests completed.")