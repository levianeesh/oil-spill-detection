# Oil Spill Detection Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)
![Model](https://img.shields.io/badge/Models-U--Net%20%7C%20DeepLabV3%2B-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A complete pipeline for **oil spill segmentation** using **U-Net** and **DeepLabV3+**, trained on satellite SAR imagery and enhanced with **AIS (Automatic Identification System) marine traffic data**. The repository contains training notebooks, an inference pipeline, and an easy-to-follow Streamlit app for quick forensic inspection of SAR images.

---

## Table of Contents

* [Project Structure](#project-structure)
* [Required Datasets](#required-datasets)
* [Installation (local)](#installation-local)
* [Run the Streamlit App (app.py)](#run-the-streamlit-app-apppy)
* [GPU Setup (optional)](#gpu-setup-optional)
* [Models Implemented](#models-implemented)
* [Training Instructions](#training-instructions)
* [Inference](#inference)
* [AIS Data Usage (Optional)](#ais-data-usage-optional)
* [How to Contribute](#how-to-contribute)
* [Do NOT Upload Large Files](#do-not-upload-large-files)
* [Contributors](#contributors)
* [License](#license)
* [Acknowledgements](#acknowledgements)

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
│   ├── 0_Prepare_AIS.py
│   ├── 1_UNet_Training.ipynb        # U-Net architecture training workflow
│   ├── 2_DeepLabV3_Training.ipynb   # DeepLabV3+ training notebook
│   └── 3_Final_Inference.ipynb      # Inference pipeline + evaluation
│
├── app.py                   # Streamlit app for quick forensic analysis
├── requirements.txt         # Python dependencies for the project
└── README.md
```

---

## 📥 Required Datasets

### 1. Oil Spill Images (Satellite SAR Data)

**Source:** [https://www.kaggle.com/datasets/nabilsherif/oil-spill](https://www.kaggle.com/datasets/nabilsherif/oil-spill) (or equivalent)

**Action:**

1. Download the dataset and unzip.
2. Place imagery into:

```
data/train/
data/test/
```

> The repository does NOT include the dataset itself — download it locally and keep it out of the repo.

### 2. AIS Vessel Data (Ship Tracking)

**Source:** [https://hub.marinecadastre.gov/pages/vesseltraffic](https://hub.marinecadastre.gov/pages/vesseltraffic) (or similar AIS providers)

**Action:**

1. Download AIS CSVs.
2. Place CSV(s) into:

```
data/ais_data/
```

---

## Installation (local)

**Recommended:** use a Python virtual environment.

```bash
# Clone the repo (if you haven't already)
git clone https://github.com/Spectrae/oil-spill-detection.git
cd oil-spill-detection

# Create and activate a virtual environment (recommended path .venv)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
---

## Run the Streamlit App (app.py)

Run the app through Streamlit's CLI (do NOT run with `/bin/python3 app.py`). Example:

```bash
cd ~/oil-spill-detection
source .venv/bin/activate      # ensure you're using the project's venv

pip install -r requirements.txt    # only first time or when updating deps

# If requirements already installed, ensure Streamlit is present
pip install streamlit

# Launch the Streamlit app
streamlit run app.py
```

Open the URL printed in terminal (usually `http://localhost:8501`) to use the UI.

**Notes & common issues**

* If you see `ModuleNotFoundError: No module named 'streamlit'`, most likely you ran the file with system Python or didn't activate the venv. Activate the venv and run `streamlit run app.py`.
* If the app throws `Error loading model: [Errno...]` make sure model weights exist in `saved_models/` (e.g. `saved_models/unet_oil_spill.h5`) or update the path in `app.py`.
* The app expects `data/ais_data/vessel_data.csv` for AIS-related functionality — ensure that file exists if you rely on AIS features.

---

## GPU Setup (optional for training)

**Recommended versions (tested):**

| Component  | Version      |
| ---------- | ------------ |
| Python     | 3.9 / 3.10   |
| TensorFlow | 2.10         |
| CUDA       | 11.2 or 11.8 |
| cuDNN      | 8.x          |

Steps:

1. Install NVIDIA drivers and verify `nvidia-smi`.
2. Install CUDA toolkit and verify `nvcc --version`.
3. Install cuDNN and copy libraries into your CUDA folder.
4. Inside venv: `pip install tensorflow==2.10`.

Test GPU in Python:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## Models Implemented

* **U-Net** — robust baseline for segmentation
* **DeepLabV3+** — stronger architecture using ASPP for multi-scale context

Trained weights should be saved to `saved_models/` as `.h5` files.

---

## Training Instructions

Place datasets into the expected folders:

```
data/train/
data/test/
data/ais_data/
```

Open the training notebooks in `notebooks/` and run them in sequence. Training notebooks will save model weights to `saved_models/`.

---

## Inference

Open `notebooks/3_Final_Inference.ipynb` to run the inference pipeline. The notebook performs preprocessing, runs the model, computes IoU/Dice scores, and visualizes mask overlays.

---

## AIS Data Usage (Optional)

Using AIS data helps to:

* Track ships near detected spill locations
* Identify suspicious vessel movements
* Correlate spill likelihood with ship activity

Place AIS CSV files under `data/ais_data/` and adapt `notebooks/0_Prepare_AIS.py` if your CSV schema differs from the expected format.

---

## How to Contribute

1. Fork the repo on GitHub.
2. Clone your fork:

```bash
git clone https://github.com/YOUR-USERNAME/oil-spill-detection.git
cd oil-spill-detection
```

3. Create a branch:

```bash
git checkout -b feature/your-feature
```

4. Add & commit changes:

```bash
git add .
git commit -m "Describe your changes"
```

5. Push & open a Pull Request:

```bash
git push origin feature/your-feature
```

Submit a PR from your fork to the `Spectrae/oil-spill-detection` repository.

---

## Do NOT Upload Large Files

Do **NOT** push the following to GitHub:

* `data/train/`
* `data/test/`
* `data/ais_data/`
* `.csv`, `.jpg`, `.png`, `.tif` (large imagery)
* `.h5` model weights (large files)

Keep all large files out of the repository and add them to `.gitignore`.

---

## Contributors

| Name             | Role                    | GitHub                                                           |
| ---------------- | ----------------------- | -----------------------------------------------------------------|
| **Rick Mondal**  | Backend Developer       | [https://github.com/Spectrae](https://github.com/Spectrae)       |
| **Aneesh Ghosh** | Research / Model Tuning | [https://github.com/levianeesh](https://github.com/levianeesh)   |
| **Ritika Kundu** | Testing & Documentation | [https://github.com/ritikakundu](https://github.com/ritikakundu) |
| **Shradha Gupta**| Data Cleaning / AIS     | [https://github.com/sgxlzel](https://github.com/sgxlzel)         |

(Feel free to add new contributors & GitHub links.)

---

## License

This project is licensed under the **MIT License**.

---

## Acknowledgements

* Kaggle SAR Dataset
* DeepLabV3+ (Google Research)
* NOAA / Marine Cadastre AIS Dataset

---

**Commands quick reference**

```bash
cd ~/oil-spill-detection
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
