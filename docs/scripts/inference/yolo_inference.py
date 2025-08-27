# -*- coding: utf-8 -*-
"""
@author: Olivier Rukundo, Ph.D.
University of Eastern Finland

"""

from ultralytics import YOLO
import os

model = YOLO(r" ")

# Input and output folders
input_folder = r" "
output_folder = r" "
os.makedirs(output_folder, exist_ok=True)

# Thresholding
conf = 0.16

# Run inference 
for fname in os.listdir(input_folder):
    if fname.lower().endswith(".png"):
        img_path = os.path.join(input_folder, fname)
        results = model(img_path, conf=conf) 
        result = results[0]

        # Save predictions
        save_path = os.path.join(output_folder, fname)
        result.save(filename=save_path)
        print(f"Saved: {save_path}")
