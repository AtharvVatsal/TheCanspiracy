from ultralytics import YOLO

model = YOLO("best.pt")

# Video file path
video_path = "test_video.mp4"  # Change this to your video
output_video = "output_video.mp4"

# Run YOLOv8 on the video
results = model.predict(source=video_path, save=True, show=True)
