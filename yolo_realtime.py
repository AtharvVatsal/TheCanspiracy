import cv2
import threading
from ultralytics import YOLO

model = YOLO("best.pt")  # Load trained YOLO model
camChoice = input("1 for webcamera\n2 for external:\n")

if camChoice == "1":
    cap = cv2.VideoCapture(0)  # Laptop webcam
elif camChoice == "2":
    droidcam_url = "[Device IP]"
    cap = cv2.VideoCapture(droidcam_url)  # DroidCam feed
else:
    print("Invalid choice. Exiting.")
    exit()

if not cap.isOpened():
    print("Error: Unable to open camera feed. Check the source.")
    exit()

native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("YOLOv8 Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Detection", native_width, native_height)

frame = None
lock = threading.Lock()

def capture_frames():
    global frame
    while True:
        ret, new_frame = cap.read()
        if ret:
            with lock:
                frame = new_frame

# Start separate thread for video capture (reduces lag)
thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

while True:
    with lock:
        if frame is None:
            continue
        frame_copy = frame.copy()

    new_width = min(1280, native_width)
    scale_factor = new_width / native_width
    new_height = int(native_height * scale_factor)
    resized_frame = cv2.resize(frame_copy, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    results = model(resized_frame)

    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box[:4])
            label = f"{model.names[int(cls)]}: {conf:.2f}"
            
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()