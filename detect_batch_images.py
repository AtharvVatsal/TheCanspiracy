from ultralytics import YOLO
import glob

model = YOLO("best.pt")

# Folder with images
image_folder = "images/"  # Change to your folder path
image_paths = glob.glob(image_folder + "*.jpg")

# Run detection on each image
for image_path in image_paths:
    results = model(image_path, show=True)
    for i, result in enumerate(results):
        result.save(f"output_{i}.jpg")
