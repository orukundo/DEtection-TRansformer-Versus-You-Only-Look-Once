# Datasets

These datasets were used to train **Detection Transformer (DETR)** and **You Only Look Once (YOLO)** for adenovirus detection in **mini-TEM (Transmission Electron Microscopy)** images.  Training was performed on the **UEF Bioinformatics Center’s High-Performance Computing (HPC) cluster**:  


---

## A) Datasets with ground truth

### YOLOv8
- **Contents:**

  - dataset.yaml
  - train/ (images + YOLO .txt labels)
  - val/ (images + YOLO .txt labels)
  - test/ (images + YOLO .txt labels)
 
- **Image size:** 640×640  
- **Label format:**  `<class_id> <x_center> <y_center> <width> <height>`
- **Class:** `adenovirus`

---

### DETR 
- **Contents:**
  - annotations/ (instances_train.json, instances_val.json, instances_test.json)
  - train/ (images)
  - val/ (images)
  - test/ (images)

- **Image size:** 640×640  
- **Label format:** COCO JSON with bounding boxes `[x, y, width, height]` and category ID  
- **Class:** `adenovirus`

---

## B) Dataset without ground truth

Used for Inference.

- **Contents:**
  - 20 images (PNG)
  - no labels or annotations
  - Image size: 2048×2048  

## Summary

| Dataset                 | Format | Image size | Labels | Purpose            |
|--------------------------|--------|------------|--------|--------------------|
| Labeled set (YOLO format) | `.txt` | 640×640    | Yes    | Evaluation (YOLO)  |
| Labeled set (COCO format) | `.json`| 640×640    | Yes    | Evaluation (DETR)  |
| Unlabeled set             | raw images | 2048×2048 | No     | Inference (both models) |

