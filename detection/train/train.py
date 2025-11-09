from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pretrained YOLOv8 nano model
    model = YOLO("yolov8s.pt")

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data="C:/Users/lucas/OneDrive/Área de Trabalho/freiburg-groceries-filtered/data.yaml",  # Path to dataset configuration file (filtered)
        epochs=50,  # Number of training epochs
        imgsz=400,
        batch=32,
        workers=8, # Image size for training
        device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )

    # Evaluate the model's performance on the validation set
    metrics = model.val()
    print(metrics)
