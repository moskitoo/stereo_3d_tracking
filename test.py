import torch
from torchvision import transforms
from ultralytics import YOLO
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"The code will run on {'GPU' if torch.cuda.is_available() else 'CPU'}.")

# Define directories for frames and labels
frames_dir = {
    (1, 2): "data/34759_final_project_rect/seq_01/image_02/data",
    (1, 3): "data/34759_final_project_rect/seq_01/image_03/data",
    (2, 2): "data/34759_final_project_rect/seq_02/image_02/data",
    (2, 3): "data/34759_final_project_rect/seq_02/image_03/data",
    (3, 2): "data/34759_final_project_rect/seq_03/image_02/data",
    (3, 3): "data/34759_final_project_rect/seq_03/image_03/data",
}

labels_paths = {
    1: "data/34759_final_project_rect/seq_01/labels.txt",
    2: "data/34759_final_project_rect/seq_02/labels.txt",
}

class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.image_paths = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(('png', 'jpg', 'jpeg'))
        ]
        # Sort image paths numerically based on the numeric part of the file names
        self.image_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Parameters
sequence_number = 3
camera_number = 2
image_dir = frames_dir[(sequence_number, camera_number)]

# Transformations
transform = transforms.Compose([
    # transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = ImageDataset(image_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# trained model path
model_path = "/home/moskit/dtu/stereo_3d_tracking/job2/runs/detect/train/weights/best.pt"

# YOLO model
model = YOLO(model_path).to(device)
model.eval()

# # Output directory
output_dir = "./test_imgs"
os.makedirs(output_dir, exist_ok=True)

# # Results processing

for images, paths in tqdm(dataloader):
    images = images.to(device)
    for index in range(images.shape[0]):
        image = images[index].unsqueeze(0)  # Add batch dimension
        image.requires_grad = True
        
        # Forward pass through YOLO model
        results = model.predict(
            task='detect',
            source= paths,
            imgsz=640,
            save=False,
            save_txt=False
        )

        # print(len(results))
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs

            # print(f"bboxes: {boxes}")

            # print(f"confidences: {boxes.conf}")
            # print(f"results: {boxes.cls}")
        
            # # if probs is not None:
            # # print(f"probs: {len(probs)}")
            # print(f"probs: {probs}")
            for id, [bbox, obj_type] in enumerate(zip(result.boxes.xyxy, result.boxes.cls)):
                print(id)
                print(f"\n bbox")
                print(bbox)
                print(f"\n obj tpye")
                print(obj_type.item())
            
            result.show()  # display to screen

torch.cuda.empty_cache()
