import cv2
import pandas as pd

from frame_manager import *

frame_number = 1
sequence_number = 1

frame_1 = get_frame(frame_number, sequence_number, 2)
frame_2 = get_frame(frame_number, sequence_number, 3)
bboxes = get_bboxes(frame_number, sequence_number)