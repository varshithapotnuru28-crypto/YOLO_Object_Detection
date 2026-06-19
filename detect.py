from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/yolov8_project-2/weights/best.pt")

# Detect objects in image
results = model.predict(
    source="test.jpg",
    save=True,
    conf=0.5
)

print("Detection Completed!")