import cv2
import pandas as pd
import os

# Define directories for frames and labels (only seq_1 and seq_2)
frames_dir = {
    (1, 2): "data/34759_final_project_rect/seq_01/image_02/data",
    (1, 3): "data/34759_final_project_rect/seq_01/image_03/data",
    (2, 2): "data/34759_final_project_rect/seq_02/image_02/data",
    (2, 3): "data/34759_final_project_rect/seq_02/image_03/data"
}

labels_paths = {
    1: "data/34759_final_project_rect/seq_01/labels.txt",
    2: "data/34759_final_project_rect/seq_02/labels.txt"
}

# Function to generate the correct image filename based on sequence
def get_frame_filename(frame_number, seq_num):
    if seq_num == 1:
        return f"{frame_number:06}.png"  # 6-digit filenames for seq_01
    elif seq_num == 2:
        return f"{frame_number:010}.png"  # 10-digit filenames for seq_02
    else:
        raise ValueError("Invalid sequence number")

# Read and sort labels for each sequence
def read_labels(labels_path):
    column_names = [
        "frame", "track_id", "type", "truncated", "occluded", "alpha",
        "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
        "dim_height", "dim_width", "dim_length",
        "loc_x", "loc_y", "loc_z", "rotation_y", "score"
    ]
    return pd.read_csv(labels_path, sep=" ", names=column_names).sort_values("frame")

# Load labels for each sequence
labels = {seq_num: read_labels(path) for seq_num, path in labels_paths.items()}

# Define colors for each class (Car, Pedestrian, Cyclist)
color_map = {
    "Car": (0, 0, 255),        # Red for Car
    "Pedestrian": (0, 255, 0), # Green for Pedestrian
    "Cyclist": (255, 0, 0)     # Blue for Cyclist
}

# Function to retrieve the frame image
def get_frame(frame_number, seq_num, camera):
    frame_filename = get_frame_filename(frame_number, seq_num)
    frame_path = os.path.join(frames_dir[(seq_num, camera)], frame_filename)
    frame = cv2.imread(frame_path)
    
    if frame is None:
        raise FileNotFoundError(f"Frame {frame_path} not found.")
    
    return frame

# Function to retrieve bounding boxes for the specified frame
def get_bboxes(frame_number, seq_num):
    seq_labels = labels[seq_num]
    frame_labels = seq_labels[seq_labels["frame"] == frame_number]
    return frame_labels

def visualize_frame(frame_number, seq_num, camera=2):
    # Initialize the video window
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    
    # Get frame and bounding boxes separately
    frame = get_frame(frame_number, seq_num, camera)
    frame_labels = get_bboxes(frame_number, seq_num)
    
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
    # Example usage: visualize a frame from sequence 1, camera 2
    visualize_frame(frame_number=50, seq_num=1, camera=2)
