# -*- coding: utf-8 -*-
"""
@author: Olivier Rukundo, Ph.D.
University of Eastern Finland

DETR: https://github.com/facebookresearch/detr
"""

import sys
sys.path.insert(0, " ")
sys.path.insert(0, " ")
from detr.models.detr import build as build_detr
from detr.main import main as detr_main
from detr.datasets.coco import make_coco_transforms
from detr.datasets.coco import CocoDetection
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import torch.serialization
import matplotlib.patches as patches

class CocoDetectionRGB(CocoDetection):
    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        if isinstance(img, Image.Image):
            img = img.convert("RGB")
        return img, target

class Args:
    def __init__(self, dataset_root, output_dir_path, num_classes_in):
        self.coco_path = dataset_root
        self.output_dir = output_dir_path
        self.num_classes = num_classes_in
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
        self.batch_size = 8
        self.epochs = 150
        self.lr_drop = 100
        self.lr = 1e-5
        self.weight_decay = 1e-4
        self.clip_max_norm = 0.1
        self.device = 'cuda'
        self.seed = 42
        self.resume = ' '
        self.start_epoch = 0
        self.eval = False
        self.frozen_weights = None
        self.set_cost_class = 1
        self.set_cost_bbox = 7
        self.set_cost_giou = 3
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 3
        self.eos_coef = 0.05
        self.distributed = False
        self.sgd = False
        self.cache_mode = False
        self.remove_difficult = False
        self.num_workers = 4
        self.world_size = 1
        self.rank = 0
        self.dist_url = 'env://'
        self.dataset_file = 'coco'
        self.lr_backbone = 1e-5
        self.dilation = False
        self.num_queries = 100


if __name__ == "__main__":
    # === CONFIG ===
    
    coco_dataset_path = ' '    
    output_path = ' '
    
    class_names = ["adenovirus"] 
    os.makedirs(output_path, exist_ok=True)

    
    def run_detr_training(dataset_root, output_dir_path, num_classes_in):
        args = Args(dataset_root, output_dir_path, num_classes_in)  
        
        detr_main(args)

    run_detr_training(coco_dataset_path, output_path, len(class_names))
    

pred_vis_dir = os.path.join(output_path, 'predictions')
os.makedirs(pred_vis_dir, exist_ok=True)

torch.serialization.add_safe_globals([ Args])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = Args(coco_dataset_path, output_path, num_classes_in=len(class_names))

model, _, postprocessors = build_detr(args)
checkpoint = torch.load(os.path.join(output_path, 'checkpoint.pth'), map_location=device)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

transform = make_coco_transforms("val")
ann_file = os.path.join(coco_dataset_path, "annotations", "instances_test2017.json")
test_img_dir = os.path.join(coco_dataset_path, 'test2017')
dataset = CocoDetectionRGB(test_img_dir, ann_file, transforms=transform, return_masks=args.masks) 
dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: list(zip(*x)), num_workers=0)

for idx, (samples, targets) in enumerate(dataloader):
    samples = samples[0].unsqueeze(0).to(device)
    outputs = model(samples)
    prob = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    boxes = outputs['pred_boxes'][0]
    scores, labels = prob.max(-1)
    keep = scores > 0.5

    img_info = dataset.coco.loadImgs(targets[0]['image_id'].item())[0]
    img_path = os.path.join(test_img_dir, img_info['file_name'])
    img_pil = Image.open(img_path).convert("RGB")
    img_draw = img_pil.copy()

    fig, ax = plt.subplots(1)
    ax.imshow(img_draw)

    for score, box, label in zip(scores[keep], boxes[keep], labels[keep]):
        box = box.detach().cpu().numpy()
        label = label.item()
        score = score.item()
        cx, cy, w, h = box
        x = cx - w / 2
        y = cy - h / 2

        rect = patches.Rectangle((x * img_pil.width, y * img_pil.height),
                                 w * img_pil.width, h * img_pil.height,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x * img_pil.width, y * img_pil.height - 10,
                f"adenovirus {score:.2f}", color='white', fontsize=8,
                bbox=dict(facecolor='red', alpha=0.5))

    label_file = os.path.join(output_path, 'predicted_labels.txt')
    with open(label_file, 'a') as f:
        for score, box, label in zip(scores[keep], boxes[keep], labels[keep]):
            cx, cy, w, h = box.tolist()
            f.write(f"{label} {score:.4f} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")

    plt.axis('off')
    save_path = os.path.join(pred_vis_dir, f"pred_{idx}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print("All predictions saved.")
