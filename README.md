# Drone Human Detection 🚁👤
### YOLOv11-Based Human Detection in Aerial Imagery

[![YOLOv11](https://img.shields.io/badge/YOLO-v11-blue)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![VisDrone](https://img.shields.io/badge/Dataset-VisDrone2019--DET-green)](http://www.aiskyeye.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🎯 Project Overview

**Drone Human Detection** focuses on detecting **humans (pedestrians and people)** in aerial images captured by drones using the **YOLOv11 object detection model** trained on the **VisDrone2019-DET** dataset.

The project builds a complete pipeline:
- Convert VisDrone annotations to YOLO format
- Create a **1-class person-only dataset**
- Train and evaluate a YOLOv11 model
- Perform inference and optional challenge submission formatting

This work is suitable for **surveillance, smart cities, traffic monitoring, and UAV-based analytics**.

---

![Detection Results](https://raw.githubusercontent.com/IamWei18/runs/detect/predict/0000272_01500_d_0000004.jpg)

---

## ✨ Key Features

### 🧍 Human Detection
- Single-class **person detection** (merged *pedestrian* + *people*)
- Optimized for **small objects** in aerial imagery

### ⚡ YOLOv11 Training Pipeline
- Uses **Ultralytics YOLOv11**
- Supports GPU acceleration
- High-resolution training (`imgsz=960`) for improved accuracy

### 🔄 Dataset Processing
- Automatic conversion from VisDrone → YOLO format
- Clean dataset splitting (train / val / test)

### 📊 Evaluation & Inference
- Validation with mAP, precision, recall
- Batch inference on images, videos, or folders
- Optional export to ONNX for deployment

---

## 🏗️ Project Structure
```plaintext
MS1 - Drone Human Detection/
├── datasets/
│   ├── VisDroneYOLO/                 # Converted 10-class YOLO dataset
│   └── VisDronePerson/               # Derived 1-class (person-only) dataset
├── scripts/
│   ├── convert_visdrone_to_yolo.py   # Step 1: Convert VisDrone to YOLO format
│   ├── make_visdrone_person.py       # Step 2: Create 1-class dataset
│   ├── train_visdrone_person.py      # Step 4: YOLOv11 training script
│   └── preds_to_visdrone_det.py      # (Optional) Convert predictions to VisDrone DET format
├── runs/                             # YOLO outputs (auto-generated, usually gitignored)
├── VisDronePerson.yaml               # Dataset config file
└── README.md
```

## 🛠️ Prerequisites

Before running the project, install the required dependencies:

```bash
pip install ultralytics torch pillow
```

> ⚠️ **GPU with CUDA is strongly recommended for training.**

---

## 🚀 Step-by-Step Workflow

### 🔹 Step 1: Convert VisDrone Annotations to YOLO Format

Converts VisDrone annotations from:

```
bbox_left, bbox_top, bbox_width, bbox_height
```

to YOLO format:

```
class_id x_center y_center width height (normalized)
```

**Script**
```bash
python scripts/convert_visdrone_to_yolo.py
```

**Output**
```
datasets/VisDroneYOLO/
├── images/train
├── images/val
├── images/test
├── labels/train
├── labels/val
└── labels/test
```

---

### 🔹 Step 2: Create a 1-Class Person Dataset

Filters the original 10-class dataset and keeps only:

- pedestrian
- people

Merged into:

- **class 0 → person**

**Script**
```bash
python scripts/make_visdrone_person.py
```

**Output**
```
datasets/VisDronePerson/
```

---

### 🔹 Step 3: Dataset YAML Configuration

**File:** `VisDronePerson.yaml`

```yaml
path: path/to/VisDronePerson
train: images/train
val: images/val
test: images/test

names:
  0: person
```

---

### 🔹 Step 4: Train YOLOv11 Model

Train the YOLOv11 nano model (`yolo11n`) on the person-only dataset.

**Script**
```bash
python scripts/train_visdrone_person.py
```

**Output**
```
runs/detect/visdrone_person_v11n/
├── weights/
│   ├── best.pt
│   └── last.pt
```

---

### 🔹 Step 5: Validate the Model

Evaluate performance on the validation set.

```python
from ultralytics import YOLO

model = YOLO("runs/detect/visdrone_person_v11n/weights/best.pt")
metrics = model.val()
print(metrics)
```

---

### 🔹 Step 6: Run Inference

Run detection on images, videos, or directories.

```python
from ultralytics import YOLO

model = YOLO("runs/detect/visdrone_person_v11n/weights/best.pt")

model.predict(
    source="path/to/images_or_video",
    conf=0.25,
    imgsz=960,
    save=True
)
```

Annotated results are saved automatically.

---

### 🔹 Step 7: Export Model (Optional)

Export for deployment (ONNX, TorchScript, etc.).

```python
from ultralytics import YOLO

model = YOLO("runs/detect/visdrone_person_v11n/weights/best.pt")
model.export(format="onnx")
```

---

### 🔹 Step 8: Convert Predictions to VisDrone Format (Optional)

For official VisDrone challenge submission.

```bash
python scripts/preds_to_visdrone_det.py
```

**Output**
```
VisDrone-compliant .txt files for evaluation
```

---

## 📊 Dataset Information

- **Dataset:** VisDrone2019-DET  
- **Task:** Object Detection (Task 1)  
- **Original Classes:** 10  
- **Project Classes:** 1 (Person)  
- **Training Images:** 6,471  
- **Validation Images:** 548  
- **Test-Dev Images:** 1,580  

---

## ⚙️ Training Configuration

| Parameter            | Value        |
|----------------------|--------------|
| Model                | YOLOv11n     |
| Image Size           | 960 × 960    |
| Batch Size           | 8            |
| Epochs               | 50           |
| Device               | GPU (CUDA)   |
| Pretrained Weights   | yolo11n.pt   |

---

## 📝 Notes

- **Small Object Challenge:** Aerial images contain small humans; higher resolution (`imgsz=960`) significantly improves results.
- **Memory Constraints:** Reduce batch size if CUDA out-of-memory occurs.
- **Evaluation:** Official evaluation requires VisDrone MATLAB script (`evalDET.m`).
- **Version Control:** The `runs/` directory and large `.pt` files are usually excluded from Git.

---

## 🔗 References

- 🌐 **VisDrone Dataset:** http://www.aiskyeye.com/  
- 📘 **Ultralytics YOLO Docs:** https://docs.ultralytics.com/

---

## 👤 Maintainer

Wei  
GitHub: https://github.com/IamWei18
