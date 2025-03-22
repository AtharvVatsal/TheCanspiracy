from ultralytics import YOLO
import cv2

model = YOLO("best.pt")  # Ensure best.pt is in the same folder

# Image path
image_path = "test.jpg"  # Replace with your image

results = model(image_path, show=True)

# Save output image
for i, result in enumerate(results):
    result.save(f"output_{i}.jpg")
