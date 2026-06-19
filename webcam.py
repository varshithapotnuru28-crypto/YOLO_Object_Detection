from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/yolov8_project-2/weights/best.pt")

# Start webcam detection
model.predict(
    source=0,
    show=True,
    conf=0.5
)