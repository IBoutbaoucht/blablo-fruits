# Blablo-Fruits LDC: Hybrid AI for Crop Disease Diagnosis

> **Competition Entry:** Local Domotics Competition (LDC) - Green Tech Theme.
> **Status:** Prototype Completed.

### 📌 Overview

Blablo-Fruits is a computer vision system designed to help farmers identify crop diseases and nutritional deficiencies in real-time. Unlike traditional classifiers that fail in messy environments, AgroScan uses a **Hybrid Architecture** that combines Object Detection with deep classification to handle complex greenhouse backgrounds.

### 🧠 The Hybrid Architecture

Our system addresses the limitations of single-model approaches by chaining two state-of-the-art models:

1. **The (YOLOv8 Nano):**
* **Role:** Scans the full image to locate leaves and fruits in real-time.
* **Training:** Fine-tuned on the PlantDoc dataset (30 classes).
* **Benefit:** Ignores background noise (dirt, pipes, sky) and focuses on the plant parts.


2. **The (EfficientNetB0):**
* **Role:** Takes the cropped leaf from YOLO and performs a deep pathological analysis.
* **Training:** Trained on a specialized 8-class dataset (Tomato/Pepper diseases).
* **Benefit:** Higher sensitivity to subtle texture changes (e.g., distinguishing *Magnesium Deficiency* from *Nitrogen Deficiency*).



### 🏷️ Capabilities

The system detects and diagnoses 8 specific conditions with high precision:

* ✅ Bacterial Disease
* ✅ Fungal Disease
* ✅ Viral Disease
* ✅ Insect Damage
* ✅ Magnesium Deficiency
* ✅ Nitrogen Deficiency
* ✅ Potassium Deficiency
* ✅ Healthy

### 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/IBoutbaoucht/blablo-fruits.git
cd blablo-fruits

# Install dependencies
pip install -r requirements.txt

```

### 🚀 Usage

Place your image in the folder and run:

```bash
python main.py --image test_leaf.jpg

```

**Sample Output:**

```text
[INFO] Analyzing image...
--- 🔍 DIAGNOSIS REPORT ---
Object 1:
  • Diagnosis: fungal_disease (Verified)
  • Confidence: 0.94
  • Location: [50, 100, 200, 300]

```

### 📂 Project Structure

* `/models`: Contains the trained `.pt` (YOLO) and `.keras` (EfficientNet) weights.
* `/src`: Modular python code for the inference pipeline.
* `/notebooks`: Original research and training logs (data conversion, training loops).

### 🔬 Challenges & Findings

* **Data Imbalance:** Initial training showed bias towards "Healthy" classes. We mitigated this by using `ImageDataGenerator` for augmentation during the EfficientNet training phase.
* **Localization:** Standalone classifiers failed when multiple leaves were present. The YOLO integration improved detection accuracy in crowded images by isolating the region of interest.

---
