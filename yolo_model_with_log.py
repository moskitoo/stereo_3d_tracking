import logging
from ultralytics import YOLO
import os


# Setup logging
log_path = '/zhome/c7/f/213256/pfas/job2/hpc_training.log'
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite log file on each run
)
logger = logging.getLogger()

def log_results(results):
    """
    Log the relevant metrics from YOLO training and validation results.
    """
    if hasattr(results, 'box'):
        logger.info(f"mAP50: {results.box.map50:.3f}")
        logger.info(f"mAP50-95: {results.box.map:.3f}")
        logger.info(f"Mean Precision: {results.box.mp:.3f}")
        logger.info(f"Mean Recall: {results.box.mr:.3f}")
        logger.info("Per-class metrics:")
        for i, class_name in results.names.items():
            try:
                precision, recall, ap50, ap95 = results.box.class_result(i)
                logger.info(f"Class '{class_name}': Precision={precision:.3f}, Recall={recall:.3f}, AP50={ap50:.3f}, AP50-95={ap95:.3f}")
            except IndexError:
                logger.info(f"No metrics available for class '{class_name}' (index {i}).")

try:
    # Start logging
    logger.info("Starting YOLOv8 training...")

    # Load a COCO-pretrained YOLOv8n model
    model = YOLO("yolov8x.pt")

    # Update paths to match your Google Drive setup
    data_split_yaml = '/zhome/c7/f/213256/pfas/data_split.yaml'

    # Train the model
    logger.info("Training started...")
    results_train = model.train(
        data=data_split_yaml,  # Path to your YAML file
        epochs=50,             # Number of epochs
        fraction=1,            # Fraction of the data to use
        workers=4,             # Number of workers
        imgsz=640,             # Input image size
        batch=4,               # Batch size
    )
    logger.info("Training completed.")

    # Log training metrics
    log_results(results_train)

    # Export the trained model in TorchScript format
    model_path = '/zhome/c7/f/213256/pfas/job2/best_model.torchscript'
    logger.info(f"Exporting the model to {model_path}...")
    model.export(format="torchscript", path=model_path)
    logger.info("Model exported successfully.")

    # Validate the model
    logger.info("Validation started...")
    results_val = model.val(
        data=data_split_yaml,  # Path to the YAML file for validation
        imgsz=640,             # Image size for validation
        save=True              # Save validation results
    )
    logger.info("Validation completed.")

    # Log validation metrics
    log_results(results_val)

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")
    raise
