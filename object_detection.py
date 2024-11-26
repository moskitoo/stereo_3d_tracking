from PIL import Image
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ultralytics import YOLO

from typing import List, Tuple


class ImageDataset(Dataset):
    """Custom dataset for loading image sequences with numeric sorting from left and right cameras."""
    def __init__(self, left_img_directory: str, right_img_directory: str, transform=None):
        self.left_img_directory = left_img_directory
        self.right_img_directory = right_img_directory
        self.left_image_paths, self.right_image_paths = self._get_sorted_image_paths()
        self.transform = transform

    def _get_sorted_image_paths(self) -> Tuple[List[str], List[str]]:
        """Retrieve and numerically sort image paths from both directories."""
        valid_extensions = ('png', 'jpg', 'jpeg')
        
        # Get sorted paths for the left camera
        left_paths = [
            os.path.join(self.left_img_directory, f)
            for f in os.listdir(self.left_img_directory)
            if f.endswith(valid_extensions)
        ]
        left_paths = sorted(left_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        # Get sorted paths for the right camera
        right_paths = [
            os.path.join(self.right_img_directory, f)
            for f in os.listdir(self.right_img_directory)
            if f.endswith(valid_extensions)
        ]
        right_paths = sorted(right_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        # Ensure the number of images matches and filenames align
        if len(left_paths) != len(right_paths):
            raise ValueError("Number of images in left and right directories do not match.")

        left_basenames = [os.path.basename(f) for f in left_paths]
        right_basenames = [os.path.basename(f) for f in right_paths]
        if left_basenames != right_basenames:
            raise ValueError("Filenames in left and right directories do not align.")

        return left_paths, right_paths

    def __len__(self) -> int:
        return len(self.left_image_paths)

    def __getitem__(self, idx: int):
        left_img_path = self.left_image_paths[idx]
        right_img_path = self.right_image_paths[idx]

        left_image = Image.open(left_img_path).convert("RGB")
        right_image = Image.open(right_img_path).convert("RGB")

        left_raw_img = transforms.ToTensor()(left_image.copy())
        right_raw_img = transforms.ToTensor()(right_image.copy())

        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image, left_raw_img, right_raw_img, left_img_path, right_img_path

class ObjectDetector:
    """Manages object detection using YOLO model."""
    def __init__(self, sequence_number: int = 3):
        self.sequence_number = sequence_number
        
        self.frames_dir = {
            (1, 2): "data/34759_final_project_rect/seq_01/image_02/data",
            (1, 3): "data/34759_final_project_rect/seq_01/image_03/data",
            (2, 2): "data/34759_final_project_rect/seq_02/image_02/data",
            (2, 3): "data/34759_final_project_rect/seq_02/image_03/data",
            (3, 2): "data/34759_final_project_rect/seq_03/image_02/data",
            (3, 3): "data/34759_final_project_rect/seq_03/image_03/data",
        }

        self.left_image_dir = self.frames_dir[(sequence_number, 2)]
        self.right_image_dir = self.frames_dir[(sequence_number, 3)]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}.")

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = ImageDataset(self.left_image_dir, self.right_image_dir, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        model_path = "/home/moskit/dtu/stereo_3d_tracking/job2/runs/detect/train/weights/best.pt"
        self.model = YOLO(model_path).to(self.device)
        self.model.eval()

    def detect_objects(self, image_path: str) -> List:
        """Perform object detection on an image."""
        return self.model.predict(
            task='detect',
            conf=0.4,
            source=image_path,
            imgsz=640,
            save=False,
            save_txt=False
        )