import cv2
import pandas as pd
import os

# Directories for frames and labels
frames_dir_1 = "data/34759_final_project_rect/seq_01/image_02/data"
frames_dir_2 = "data/34759_final_project_rect/seq_02/image_02/data"
frames_dir_3 = "data/34759_final_project_rect/seq_03/image_02/data"
labels_path_1 = "data/34759_final_project_rect/seq_01/labels.txt"
labels_path_2 = "data/34759_final_project_rect/seq_02/labels.txt"
labels_path_3 = "data/34759_final_project_rect/seq_03/labels.txt"

# Function to generate the correct image filename based on sequence
def get_frame_filename(frame_number, seq_num):
    if seq_num == 1:
        return f"{frame_number:06}.png"  # 6-digit filenames for seq_01
    elif seq_num == 2 or seq_num == 3:
        return f"{frame_number:010}.png"  # 10-digit filenames for seq_02 and seq_03
    else:
        raise ValueError("Invalid sequence number")

# Read the labels for both sequences
def read_labels(labels_path):
    column_names = [
        "frame", "track_id", "type", "truncated", "occluded", "alpha",
        "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
        "dim_height", "dim_width", "dim_length",
        "loc_x", "loc_y", "loc_z", "rotation_y", "score"
    ]
    return pd.read_csv(labels_path, sep=" ", names=column_names)

# Load and sort labels
labels_1 = read_labels(labels_path_1).sort_values("frame")
labels_2 = read_labels(labels_path_2).sort_values("frame")

labels = [labels_1, labels_2]
frames_dir = [frames_dir_1, frames_dir_2]

# Define colors for each class (Car, Pedestrian, Cyclist)
color_map = {
    "Car": (0, 0, 255),        # Red for Car
    "Pedestrian": (0, 255, 0), # Green for Pedestrian
    "Cyclist": (255, 0, 0)     # Blue for Cyclist
}

def visualize_frame(frame_number, seq_num):
    # Initialize the video window
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    seq_labels = labels[seq_num - 1]
    frame_filename = get_frame_filename(frame_number, seq_num)
    frame_path = os.path.join(frames_dir[seq_num - 1], frame_filename)
    frame = cv2.imread(frame_path)
    
    if frame is None:
        print(f"Frame {frame_path} not found.")
        return
    
    # Filter labels for the current frame
    frame_labels = seq_labels[seq_labels["frame"] == frame_number]
    
    # Draw each bounding box on the frame
    draw_bboxes(frame, frame_labels)
    
    # Display the frame in a single window
    cv2.imshow("Video", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_bboxes(frame, frame_labels):
    for _, row in frame_labels.iterrows():
        bbox_left, bbox_top = int(row["bbox_left"]), int(row["bbox_top"])
        bbox_right, bbox_bottom = int(row["bbox_right"]), int(row["bbox_bottom"])
        obj_type = row["type"]
        
        # Choose color based on object type
        color = color_map.get(obj_type, (255, 255, 255))  # Default to white if unknown type
        
        # Draw bounding box with the selected color
        cv2.rectangle(frame, (bbox_left, bbox_top), (bbox_right, bbox_bottom), color, 2)
        
        # Annotate with object type
        cv2.putText(frame, obj_type, (bbox_left, bbox_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Run the visualization when the script is executed directly
if __name__ == "__main__":
    visualize_frame(frame_number=5, seq_num=1)
