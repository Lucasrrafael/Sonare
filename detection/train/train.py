from ultralytics import YOLO

# Load a pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train the model on the COCO8 dataset for 100 epochs
train_results = model.train(
    data="detection/train/freiburg-groceries.v2i.yolov8/data.yaml",  # Path to dataset configuration file
    epochs=10,  # Number of training epochs
    imgsz=640,  # Image size for training
    device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)

# Evaluate the model's performance on the validation set
metrics = model.val()
print(metrics)
