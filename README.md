# YOLOv8 Real-time Object Detection

> 🔥 Turn your PC or phone camera into a real-time AI scanner!

This project implements YOLOv8-based object detection across multiple formats - from single images to real-time video feeds including webcam and mobile phone cameras via DroidCam.

![YOLOv8 Detection Banner](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png)

## 🚀 Features

- **Real-time Detection**: Process live video feeds with optimized performance
- **Multi-format Support**: Works with images, video files, and camera streams
- **DroidCam Integration**: Use your phone as a smart AI camera
- **Visual Feedback**: Display object names with confidence scores in real-time
- **Performance Optimization**: Multithreading and resolution tweaks for smooth operation
- **Data Visualization**: Comprehensive metric analysis and visualization tools

## 📁 Project Structure

```
YOLOv8-Object-Detection/
│
├── detect_batch_images.py    # Process multiple images in a batch
├── detect_image.py           # Detection on a single image
├── detect_video.py           # Detection in video files
├── yolo_detect.py            # Core YOLOv8 model loading functionality
├── yolo_metrics_analysis.py  # Data visualization and metrics analysis
├── yolo_realtime.py          # Real-time detection from camera feeds
│
├── Data Viz/                 # Data visualization outputs and utilities
│   ├── bar_charts.png        # Precision per class
│   ├── scatter_plots.png     # Precision vs Recall
│   ├── line_charts.png       # mAP50 over classes
│   ├── heatmaps.png          # Correlation between metrics
│   ├── histograms.png        # Distribution of mAP50
│   ├── box_plots.png         # Distribution of Precision per class
│   └──treemaps.png          # Instances per class
│
├── Dataset-1/                # Training dataset files
│   ├── images/               # Training images
│   ├── labels/               # Ground truth annotations
│   └── data.yaml   # Dataset configuration
│
└── YOLO-Training-Results/    # Model training outputs
    ├── weights/              # Trained model weights (.pt files)
    ├── runs/                 # Training runs and experiments
    └── metrics/              # Training and validation metrics
```

## 💻 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/TheCanspiracy.git
   cd TheCanspiracy
   ```

2. Install dependencies:
   ```bash
   pip install ultralytics opencv-python matplotlib seaborn squarify numpy pandas
   ```

3. Install DroidCam on your mobile device (optional for phone camera usage)

## 🔧 Usage

### Real-time Detection with Webcam

```bash
python yolo_realtime.py
```

When prompted, select option 1 for webcam.

### Real-time Detection with DroidCam

1. Install DroidCam on your mobile device and computer
2. Connect both devices to the same Wi-Fi network
3. Note the IP address shown in the DroidCam app
4. Run:
   ```bash
   python yolo_realtime.py
   ```
5. When prompted, select option 2 for external camera
6. The program will connect to your phone's camera feed

### Image Detection

```bash
python detect_image.py --source path/to/your/image.jpg
```

### Batch Image Processing

```bash
python detect_batch_images.py --source path/to/image/folder --output path/to/output/folder
```

### Video Processing

```bash
python detect_video.py --source path/to/video.mp4 --output path/to/output.mp4
```

### Metrics Analysis and Visualization

```bash
python yolo_metrics_analysis.py --results path/to/results/folder
```

## 📊 Data Visualization

The project includes various visualization tools to analyze detection performance:

- **Bar Charts**: Precision per class
- **Scatter Plots**: Precision vs Recall relationships
- **Line Charts**: mAP50 tracking over classes
- **Heatmaps**: Correlation visualization between metrics
- **Histograms**: Distribution analysis of mAP50
- **Box Plots**: Distribution of Precision per class
- **Treemaps**: Visualization of instances per class
- **Waterfall Charts**: Custom visualization of instances per class
- **Gantt Charts**: Simplified class-wise progress visualization

## 🔄 Model Training

To train your own YOLOv8 model:

1. Prepare your dataset in YOLO format in the Dataset-1 folder
2. Create a dataset.yaml file with your class names and paths
3. Train using:
   ```bash
   yolo task=detect mode=train model=yolov8n.pt data=Dataset-1/dataset.yaml epochs=100 imgsz=640
   ```
4. The training results will be saved in the YOLO-Training-Results folder

## ⌨️ Keyboard Controls

When using the real-time detection:

- **Q**: Quit the application
- **S**: Save the current frame
- **+/-**: Adjust confidence threshold

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- DroidCam for the mobile camera streaming capability

