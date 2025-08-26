# Datasets

These datasets were used to train **Detection Transformer (DETR)** and **You Only Look Once (YOLO)** for adenovirus detection in **mini-TEM (Transmission Electron Microscopy)** images.  

---

## A) Test set with ground truth

### YOLO format
- **Contents:**

  - dataset.yaml
  - train/ (images + YOLO .txt labels)
  - val/ (images + YOLO .txt labels)
  - test/ (images + YOLO .txt labels)
 
- **Image size:** 640×640  
- **Label format:**  
Each line → `<class_id> <x_center> <y_center> <width> <height>` (normalized 0–1)  
- **Classes:** single class of interest → `adenovirus`

---

### DETR (COCO) format
- **Contents:**
  - annotations/ (instances_train2017.json, instances_val2017.json, instances_test2017.json)
  - train2017/ (images)
  - val2017/ (images)
  - test2017/ (images)

- **Image size:** 640×640  
- **Label format:** COCO JSON with bounding boxes `[x, y, width, height]` and category ID  
- **Classes:** single class of interest → `adenovirus`

---

## B) Test set without ground truth

Used for Inference (qualitative comparison only).

- **Contents:**
20 images (PNG/JPG)
no labels or annotations

- **Image size: 2048×2048  
- **Purpose:** Compare how YOLO and DETR generalize to unseen adenovirus mini-TEM images without ground truth  

## Summary

| Dataset                 | Format | Image size | Labels | Purpose            |
|--------------------------|--------|------------|--------|--------------------|
| Labeled set (YOLO format) | `.txt` | 640×640    | Yes    | Evaluation (YOLO)  |
| Labeled set (COCO format) | `.json`| 640×640    | Yes    | Evaluation (DETR)  |
| Unlabeled set             | raw images | 2048×2048 | No     | Inference (both models) |

