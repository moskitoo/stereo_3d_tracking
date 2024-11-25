import torch
from torchvision import transforms
from ultralytics import YOLO
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from feature_manager_v2 import *


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
        raw_img = transforms.ToTensor()(image.copy())

        if self.transform:
            image = self.transform(image)
        return image, raw_img, img_path


class ObjectDetector:
    def __init__(self) -> None:
        self.sequence_number = 3
        self.camera_number = 2

        self.frames_dir = {
            (1, 2): "data/34759_final_project_rect/seq_01/image_02/data",
            (1, 3): "data/34759_final_project_rect/seq_01/image_03/data",
            (2, 2): "data/34759_final_project_rect/seq_02/image_02/data",
            (2, 3): "data/34759_final_project_rect/seq_02/image_03/data",
            (3, 2): "data/34759_final_project_rect/seq_03/image_02/data",
            (3, 3): "data/34759_final_project_rect/seq_03/image_03/data",
        }

        self.labels_paths = {
            1: "data/34759_final_project_rect/seq_01/labels.txt",
            2: "data/34759_final_project_rect/seq_02/labels.txt",
        }

        self.image_dir = self.frames_dir[(self.sequence_number, self.camera_number)]

        self.transform = transforms.Compose([
            # transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])

        self.dataset = ImageDataset(self.image_dir, transform=self.transform)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"The code will run on {'GPU' if torch.cuda.is_available() else 'CPU'}.")

        self.model_path = "/home/moskit/dtu/stereo_3d_tracking/job2/runs/detect/train/weights/best.pt"


        self.model = YOLO(self.model_path).to(self.device)
        self.model.eval()

class ObjectTracker:
    def __init__(self) -> None:
        self.object_detector = ObjectDetector()
        self.object_container = []

    def run(self):
        while True:
            # print(f"frame number: {current_frame_number}")

            # frame_1 = get_frame(current_frame_number, sequence_number, 2)

            for image, raw_image, path in tqdm(self.object_detector.dataloader):
                image = image.to(self.object_detector.device)

                raw_image = raw_image.squeeze(0)
                raw_image = raw_image.permute(1, 2, 0).numpy()
                raw_image = (raw_image * 255).astype(np.uint8) 

                image.requires_grad = True
                
                # Forward pass through YOLO model
                detection_output = self.object_detector.model.predict(
                    task='detect',
                    source=path,
                    imgsz=640,
                    save=False,
                    save_txt=False
                )

                # print(len(results))
                # for result in results:
                #     boxes = result.boxes  # Boxes object for bounding box outputs
                #     masks = result.masks  # Masks object for segmentation masks outputs
                #     keypoints = result.keypoints  # Keypoints object for pose outputs
                #     probs = result.probs  # Probs object for classification outputs
                #     obb = result.obb  # Oriented boxes object for OBB outputs
                #     result.show()  # display to screen

                # detection_output = get_detection_results(current_frame_number, sequence_number)

                detected_objects = detect_objects_yolo(raw_image, detection_output[0])

                frame_with_tracked_objects = visualize_objects(raw_image, self.object_container)

                self.object_container, matches = match_objects(detected_objects, self.object_container)

                frame_with_detected_objects = visualize_objects(raw_image, detected_objects)

                # frame_with_matched_objects = visualize_matched_objects(raw_image, self.object_container, detected_objects, matches)

                masked_frame_1 = get_masked_image(raw_image, detection_output)
                bbox_frame = draw_bounding_boxes(raw_image, detection_output)
                
                combined_frames = combine_frames([frame_with_tracked_objects, frame_with_detected_objects, masked_frame_1, bbox_frame])
                
                print("\n\n")

                # Display current frame number
                # cv2.putText(combined_frames, f"Frame: {current_frame_number}", (10, 30), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.namedWindow("Frame with combined_frames", cv2.WINDOW_NORMAL)
                cv2.imshow("Frame with combined_frames", combined_frames)

                # cv2.namedWindow("Frame with frame_with_matched_objects", cv2.WINDOW_NORMAL)
                # cv2.imshow("Frame with frame_with_matched_objects", frame_with_matched_objects)

                # Wait for key press
                key = cv2.waitKey(0) & 0xFF

                if key == ord('q'):  # Quit
                    break

if __name__ == '__main__':
    tracker = ObjectTracker()
    tracker.run()