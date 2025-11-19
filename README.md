# Oil Spill Detection Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)
![Model](https://img.shields.io/badge/Models-U--Net%20%7C%20DeepLabV3%2B-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A complete pipeline for **oil spill segmentation** using **U-Net** and **DeepLabV3+**, trained on satellite SAR imagery and enhanced with **AIS (Automatic Identification System) marine traffic data**.

This project includes training notebooks, an inference pipeline, and a clean modular structure for easy collaboration among team members.

---

## Project Structure

```plaintext
Oil_Spill_Project/
├── data/
│   ├── train/               # Kaggle Train dataset (images + masks)
│   ├── test/                # Kaggle Test dataset
│   └── ais_data/            # Marine Cadastre AIS ship-tracking data
│
├── saved_models/            # Trained model weights (.h5 files) are saved here
│
├── notebooks/
│   ├── 1_UNet_Training.ipynb        # U-Net architecture training workflow
│   ├── 2_DeepLabV3_Training.ipynb   # DeepLabV3+ model training notebook
│   └── 3_Final_Inference.ipynb      # Inference pipeline + evaluation
│
└── requirements.txt         # Python dependencies for the project
````

---

## 📥 Required Datasets

### **1. Oil Spill Images (Satellite SAR Data)**

**Source:** [https://www.kaggle.com/competitions/airbus-ship-detection/data](https://www.kaggle.com/competitions/airbus-ship-detection/data)
**Action:**

* Download → Unzip → place into:

```
data/train/
data/test/
```

### **2. AIS Vessel Data (Ship Tracking)**

**Source:** [https://marinecadastre.gov/ais/](https://marinecadastre.gov/ais/)
**Action:**

* Download any AIS CSV → place into:

```
data/ais_data/
```

---

## Installation

```bash
git clone https://github.com/Spectrae/oil-spill-detection.git
cd oil-spill-detection
pip install -r requirements.txt
```

---

## ⚡ GPU Setup (TensorFlow GPU + CUDA + cuDNN)

### **Recommended Versions**

| Component  | Version                                                    |
| ---------- | ---------------------------------------------------------- |
| Python     | 3.9 / 3.10                                                 |
| TensorFlow | 2.10 (last version with GPU support without NVIDIA wheels) |
| CUDA       | 11.2 or 11.8                                               |
| cuDNN      | 8.x                                                        |

---

### **1️⃣ Install NVIDIA GPU Drivers**

Check GPU:

```bash
nvidia-smi
```

If it shows GPU info → drivers are installed.

---

### **2️⃣ Install CUDA Toolkit**

Download from:
[https://developer.nvidia.com/cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)

Check installation:

```bash
nvcc --version
```

---

### **3️⃣ Install cuDNN**

Download:
[https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

Extract → copy cuDNN files into your CUDA directory:

```
/usr/local/cuda/include
/usr/local/cuda/lib64
```

---

### **4️⃣ Install TensorFlow GPU version**

```bash
pip install tensorflow==2.10
```

Test GPU recognition:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

If it prints your GPU → setup is successful.

---

## Models Implemented

### **1. U-Net**

* Strong baseline for segmentation
* Robust with limited data

### **2. DeepLabV3+**

* State-of-the-art segmentation
* Uses Atrous Spatial Pyramid Pooling (ASPP)

---

## Training Instructions

Place datasets into:

```
data/train/
data/test/
data/ais_data/
```

Open:

```
notebooks/1_UNet_Training.ipynb
notebooks/2_DeepLabV3_Training.ipynb
```

Model weights will save into:

```
saved_models/
```

---

## Inference

Run:

```
notebooks/3_Final_Inference.ipynb
```

Performs:

* Preprocessing
* Spill detection
* IoU / Dice evaluation
* Model comparison
* Mask overlays

---

## AIS Data Usage (Optional)

AIS data helps to:

* Track ships near spill locations
* Identify suspicious vessel paths
* Correlate ship activity with detected spills

---

## 🤝 How to Contribute (Fork → Clone → Branch → Commit → PR)

### **1️⃣ Fork Repo**

Go to:
[https://github.com/Spectrae/oil-spill-detection](https://github.com/Spectrae/oil-spill-detection)
Click **Fork**.

---

### **2️⃣ Clone Your Fork**

```bash
git clone https://github.com/YOUR-USERNAME/oil-spill-detection.git
cd oil-spill-detection
```

---

### **3️⃣ Create a New Branch**

```bash
git checkout -b feature-name
```

---

### **4️⃣ Add & Commit Changes**

```bash
git add .
git commit -m "Describe your update"
```

---

### **5️⃣ Push to Your Fork**

```bash
git push origin feature-name
```

---

### **6️⃣ Submit a Pull Request**

* Go to your fork
* Click **Compare & Pull Request**
* Submit the PR

---

## ⚠️ Do NOT Upload Large Files

Do **NOT** push:

* `data/train/`
* `data/test/`
* `data/ais_data/`
* `.csv`, `.jpg`, `.png`, `.tif`
* `.h5` weights

These must remain local.

---

## 👨‍💻 Contributors

| Name              | Role                            | GitHub                                                     |
| ----------------- | ------------------------------- | ---------------------------------------------------------- |
| **Rick Mondal**   | Backend Developer               | [https://github.com/Spectrae](https://github.com/Spectrae) |
| **Contributor 2** | Research / Model Tuning         | *(Add GitHub link)*                                        |
| **Contributor 3** | Data Cleaning / AIS Integration | *(Add GitHub link)*                                        |
| **Contributor 4** | Testing & Documentation         | *(Add GitHub link)*                                        |

> Add your team members' GitHub profiles as they join.

---

## License

This project is licensed under the **MIT License**.

---

## Acknowledgements

* Kaggle SAR Dataset
* DeepLabV3+ (Google Research)
* NOAA Marine Cadastre AIS Dataset
