from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/yolov8_project-2/weights/best.pt")

# Detect objects in video
results = model.predict(
    source="video.mp4",
    save=True,
    conf=0.5
)

print("Video Detection Completed!")