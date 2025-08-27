# -*- coding: utf-8 -*-
"""
@author: Olivier Rukundo, Ph.D.
University of Eastern Finland

YOLOv8: https://github.com/ultralytics/ultralytics
"""

import os
import torch
import shutil
from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box

# Device: CPU or GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset base path
base_path = ' '

# Dataset.yaml content
dataset_yaml_path = os.path.join(base_path, 'dataset.yaml')
dataset_yaml = f"""
path: {base_path}
train: train/images
val: val/images
test: test/images
names:
  0: adenovirus
"""

with open(dataset_yaml_path, 'w') as f:
    f.write(dataset_yaml)

print("dataset.yaml file created!")

# Load YOLO model
print("Loading YOLOv8 model for training...")
model = YOLO('yolov8n.pt')

# Train
print("Starting training...")
model.train(
    data=dataset_yaml_path,
    epochs=2000,
    imgsz=640,
    batch=16,
    patience=200,
    name='adenovirus_detection',
    exist_ok=True, 
    device=device,
    seed=42,
    lr0=0.01,
    optimizer='SGD',
    momentum=0.937,
    weight_decay=0.0005
)

# Save best model
best_model_src = ' '
best_model_dest = os.path.join(base_path, ' ')

print(f"Saving best model to {best_model_dest}...")
os.makedirs(os.path.dirname(best_model_dest), exist_ok=True)
shutil.copy2(best_model_src, best_model_dest)
print("Model saved!")

# Load and evaluate
print("Loading the best trained YOLOv8 model...")
model = YOLO(best_model_dest)

print("Running evaluation on test set...")
metrics = model.val(data=dataset_yaml_path,split='test')

print("Evaluation Metrics on Test Set:")
print(metrics)

# Saving images with Predictions
results = model.val(data=dataset_yaml_path, split='test', save=True, save_txt=False)

output_dir = ' '
os.makedirs(output_dir, exist_ok=True)

for i, result in enumerate(results):
    path = result.path
    im0 = result.orig_img.copy()
    boxes = result.boxes

    for j, box in enumerate(boxes):
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"
        save_one_box(xyxy, im0, file=os.path.join(output_dir, f"{i}_{j}.png"), label=label, save_crop=False)