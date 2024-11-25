
from ultralytics import YOLO
import os

# Load a COCO-pretrained YOLOv8n model
model = YOLO("yolov8x.pt")

# Update paths to match your Google Drive setup
data_split_yaml = '/zhome/c7/f/213256/pfas/data_split.yaml'  

# Train the model
results_train = model.train(
    data=data_split_yaml,  # Path to your YAML file
    epochs=50,             # Number of epochs
    fraction=1,         # Fraction of the data to use
    imgsz=640,             # Input image size
    #workers=2,             # Number of workers
    batch=4,              # Batch size
)

# Export the trained model in TorchScript format
model_path = 'zhome/c7/f/213256/pfas/best_model.torchscript'
model.export(format="torchscript", path=model_path)

# Validate using the YAML file
results = model.val(
    data=data_split_yaml,  # Path to the YAML file for validation
    imgsz=640,             # Image size for validation
    save=True             # Save validation results
)

# Access mAP50
print(f"mAP50: {results.box.map50:.3f}")

# Access mean precision and recall
print(f"Precision: {results.box.mp:.3f}")  # Mean precision
print(f"Recall: {results.box.mr:.3f}")     # Mean recall

# Access mAP across IoU thresholds (50 to 95)
print(f"mAP50-95: {results.box.map:.3f}")

# Access per-class metrics
for i, class_name in results.names.items():
    try:
        precision, recall, ap50, ap95 = results.box.class_result(i)
        print(f"Class '{class_name}': Precision={precision:.3f}, Recall={recall:.3f}, AP50={ap50:.3f}, AP50-95={ap95:.3f}")
    except IndexError:
        print(f"No metrics available for class '{class_name}' (index {i}).")


