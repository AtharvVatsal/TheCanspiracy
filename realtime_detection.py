import cv2
import threading
import time
import numpy as np
from ultralytics import YOLO

gpu_enabled = input("Enable GPU? (y/n): ").strip().lower() == 'y'

model = YOLO("best.pt")
if not gpu_enabled:
    model.to('cpu')

# Select Camera Source
camChoice = input("1 for webcam\n2 for external:\n")
if camChoice == "1":
    cap = cv2.VideoCapture(1)  # Laptop webcam
elif camChoice == "2":
    droidcam_url = "http://192.168.181.139:4747/video"
    cap = cv2.VideoCapture(droidcam_url)  # DroidCam feed
else:
    print("Invalid choice. Exiting.")
    exit()

if not cap.isOpened():
    print("Error: Unable to open camera feed. Check the source.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  

native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Detection", native_width, native_height)

frame = None
lock = threading.Lock()

fps = 0
prev_time = time.time()
obj_count = 0
conf_threshold = 50

recording = False
paused = False
detection_enabled = True
show_boxes = True
show_labels = True
show_fps = True
show_help = False

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = None

np.random.seed(42)
colors = {i: np.random.randint(100, 255, size=(3,)).tolist() for i in range(80)}

def capture_frames():
    global frame
    while True:
        if not paused:
            ret, new_frame = cap.read()
            if ret:
                with lock:
                    frame = new_frame

thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

def update_conf(val):
    global conf_threshold
    conf_threshold = val

cv2.createTrackbar("Confidence (%)", "YOLOv8 Detection", conf_threshold, 100, update_conf)

while True:
    with lock:
        if frame is None:
            continue
        frame_copy = frame.copy()

    new_width = min(1280, native_width)
    scale_factor = new_width / native_width
    new_height = int(native_height * scale_factor)
    resized_frame = cv2.resize(frame_copy, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time else 0
    prev_time = curr_time

    obj_count = 0

    if detection_enabled:
        results = model(resized_frame)
        visible_objects = {}

        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                if conf * 100 < conf_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box[:4])
                class_id = int(cls)
                label = f"{model.names[class_id]}: {conf:.2f}"
                visible_objects[model.names[class_id]] = visible_objects.get(model.names[class_id], 0) + 1
                color = colors[class_id]
                
                if show_boxes:
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), color, 2)
                
                if show_labels:
                    font_scale = 1.2
                    thickness = 2
                    cv2.putText(resized_frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_PLAIN, font_scale, color, thickness, cv2.LINE_AA)

        obj_count = sum(visible_objects.values())
    
    if show_fps:
        cv2.putText(resized_frame, f"FPS: {fps}", (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(resized_frame, f"Objects Detected: {obj_count}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    if recording:
        if out is None:
            out = cv2.VideoWriter("output.mp4", fourcc, 20.0, (new_width, new_height))
        out.write(resized_frame)
        cv2.putText(resized_frame, "Recording...", (20, 100), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2, cv2.LINE_AA)

    if show_help:
        help_text = [
            "R: Start/Stop Recording",
            "P: Pause/Resume Video Feed",
            "D: Enable/Disable Detection",
            "B: Show/Hide Bounding Boxes",
            "L: Show/Hide Labels",
            "F: Show/Hide FPS Counter",
            "H: Show/Hide Help Screen",
            "Q: Exit Application"
        ]
        for i, text in enumerate(help_text):
            cv2.putText(resized_frame, text, (20, 150 + i * 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("YOLOv8 Detection", resized_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        recording = not recording
        if not recording and out is not None:
            out.release()
            out = None
    elif key == ord('p'):
        paused = not paused
    elif key == ord('d'):
        detection_enabled = not detection_enabled
    elif key == ord('b'):
        show_boxes = not show_boxes
    elif key == ord('l'):
        show_labels = not show_labels
    elif key == ord('f'):
        show_fps = not show_fps
    elif key == ord('h'):
        show_help = not show_help
    elif key == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()