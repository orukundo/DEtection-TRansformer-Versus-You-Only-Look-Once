# -*- coding: utf-8 -*-
"""
@author: Olivier Rukundo, Ph.D.
University of Eastern Finland

"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, r" ")  
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from detr.models.detr import build as build_detr
from detr.datasets.coco import make_coco_transforms


class Args:
    def __init__(self):
        self.num_classes = 2 
        self.backbone = 'resnet50'
        self.position_embedding = 'sine'
        self.hidden_dim = 256
        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0.1
        self.nheads = 8
        self.pre_norm = False
        self.masks = False
        self.aux_loss = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataset_file = 'coco'
        self.dilation = False
        self.num_queries = 100
        self.lr_backbone = 1e-5
        self.set_cost_class = 1
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.eos_coef = 0.1

# Register Args for deserialization
torch.serialization.add_safe_globals([Args])

# Thresholding
threshold = 0.16

def load_custom_detr_model(checkpoint_path):
    args = Args()
    model, _, _ = build_detr(args)
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()
    return model, args

def infer_and_save(model, args, image_path, output_path, threshold=threshold):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_pil = Image.fromarray(image_rgb)

    print("→ image.shape (OpenCV):", image.shape)
    print("→ image_pil.size (PIL):", image_pil.size)

    transform = make_coco_transforms("val")
    image_transformed, _ = transform(image_pil, target={})
    image_transformed = image_transformed.unsqueeze(0).to(args.device)

    with torch.no_grad():
        outputs = model(image_transformed)

    prob = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    boxes = outputs['pred_boxes'][0]
    scores, labels = prob.max(-1)
    keep = scores > threshold

    img_w, img_h = image_pil.size
    dpi=100
    fig, ax = plt.subplots(1, figsize=(img_w/dpi, img_h/dpi), dpi=dpi)
    ax.imshow(image_pil)

    for score, box, label in zip(scores[keep], boxes[keep], labels[keep]):
        box_np = np.array(box.cpu())
        cx, cy, w, h = box_np
        x = (cx - w / 2) * img_w
        y = (cy - h / 2) * img_h
        w *= img_w
        h *= img_h

        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 10, f"adenovirus {score:.2f}", color='white', fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.tight_layout(pad=0)  
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()
    
    # Convert to 24-bit RGB using PIL
    with Image.open(output_path) as im:
        rgb_im = im.convert('RGB')  
        rgb_im.save(output_path)

    print(f"Saved (24-bit RGB): {output_path}")
    
## Run batch inference
input_folder = r" "
output_folder = r" "
os.makedirs(output_folder, exist_ok=True)

# Load trained model
model, args = load_custom_detr_model(r" ")

# Iterate over all images
for fname in os.listdir(input_folder):
    if fname.lower().endswith("."):
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)
        infer_and_save(model, args, in_path, out_path, threshold=threshold)
