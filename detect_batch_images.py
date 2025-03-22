from ultralytics import YOLO
import glob

model = YOLO("best.pt")

image_folder = "path/to/image"
image_paths = glob.glob(image_folder + "*.jpg")

for image_path in image_paths:
    results = model(image_path, show=True)
    for i, result in enumerate(results):
        result.save(f"output_{i}.jpg")
