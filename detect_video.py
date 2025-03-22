from ultralytics import YOLO

model = YOLO("best.pt")

# Video file path
video_path = "test_video.mp4"  # Change this to your video
output_video = "output_video.mp4"

# Run YOLOv8 on the video and ensure output is saved in MP4 format
results = model.predict(source=video_path, save=True, save_txt=False, save_conf=False, show=True)
import os
if os.path.exists("runs/predict/video"):
    os.rename("runs/predict/video", output_video)
