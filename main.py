import os
import torch
import numpy as np
import cv2
from typing import List, Tuple
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO

from feature_manager_v2 import (
    detect_objects_yolo, 
    visualize_objects, 
    match_objects, 
    visualize_matched_objects, 
    get_masked_image, 
    combine_frames
)

class ImageDataset(Dataset):
    """Custom dataset for loading image sequences with numeric sorting."""
    def __init__(self, directory: str, transform=None):
        self.directory = directory
        self.image_paths = self._get_sorted_image_paths()
        self.transform = transform

    def _get_sorted_image_paths(self) -> List[str]:
        """Retrieve and sort image paths numerically."""
        valid_extensions = ('png', 'jpg', 'jpeg')
        paths = [
            os.path.join(self.directory, f)
            for f in os.listdir(self.directory)
            if f.endswith(valid_extensions)
        ]
        return sorted(paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        raw_img = transforms.ToTensor()(image.copy())

        if self.transform:
            image = self.transform(image)
        return image, raw_img, img_path

class ObjectDetector:
    """Manages object detection using YOLO model."""
    def __init__(self, sequence_number: int = 3, camera_number: int = 2):
        self.sequence_number = sequence_number
        self.camera_number = camera_number
        
        self.frames_dir = {
            (1, 2): "data/34759_final_project_rect/seq_01/image_02/data",
            (1, 3): "data/34759_final_project_rect/seq_01/image_03/data",
            (2, 2): "data/34759_final_project_rect/seq_02/image_02/data",
            (2, 3): "data/34759_final_project_rect/seq_02/image_03/data",
            (3, 2): "data/34759_final_project_rect/seq_03/image_02/data",
            (3, 3): "data/34759_final_project_rect/seq_03/image_03/data",
        }

        self.image_dir = self.frames_dir[(sequence_number, camera_number)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}.")

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = ImageDataset(self.image_dir, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        model_path = "/home/moskit/dtu/stereo_3d_tracking/job2/runs/detect/train/weights/best.pt"
        self.model = YOLO(model_path).to(self.device)
        self.model.eval()

    def detect_objects(self, image_path: str) -> List:
        """Perform object detection on an image."""
        return self.model.predict(
            task='detect',
            source=image_path,
            imgsz=640,
            save=False,
            save_txt=False
        )

class ObjectTracker:
    """Manages object tracking across video frames."""
    def __init__(self, sequence_number: int = 3, camera_number: int = 2):
        self.object_detector = ObjectDetector(sequence_number, camera_number)
        self.object_container = []

    def process_frame(self, image, raw_image, path) -> None:
        """Process a single frame for object detection and tracking."""
        image = image.to(self.object_detector.device)
        image.requires_grad = True

        # Convert raw image for visualization
        raw_image = raw_image.squeeze(0)
        raw_image = raw_image.permute(1, 2, 0).numpy()
        raw_image = (raw_image * 255).astype(np.uint8)

        # Detect objects
        detection_output = self.object_detector.detect_objects(path)
        detected_objects = detect_objects_yolo(raw_image, detection_output[0])

        # Track objects
        frame_with_tracked_objects = visualize_objects(raw_image, self.object_container)
        self.object_container, matches = match_objects(detected_objects, self.object_container)

        # Visualization
        frame_with_detected_objects = visualize_objects(raw_image, detected_objects)
        # frame_with_matched_objects = visualize_matched_objects(raw_image, self.object_container, detected_objects, matches)
        masked_frame = get_masked_image(raw_image, detection_output[0])
        
        combined_frames = combine_frames([
            frame_with_tracked_objects, 
            frame_with_detected_objects, 
            masked_frame
        ])

        return combined_frames

    def run(self):
        """Run object tracking on all frames."""
        for image, raw_image, path in tqdm(self.object_detector.dataloader):
            combined_frames = self.process_frame(image, raw_image, path)
            
            cv2.namedWindow("Frame with combined_frames", cv2.WINDOW_NORMAL)
            cv2.imshow("Frame with combined_frames", combined_frames)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):  # Quit
                break

def main():
    tracker = ObjectTracker()
    tracker.run()

if __name__ == '__main__':
    main()