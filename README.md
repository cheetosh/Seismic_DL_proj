# Seismic Image Classification using Deep Learning

## 📌 Overview
This project focuses on classifying seismic images to detect the **presence or absence of hydrocarbons** using deep learning techniques.  
A **CNN-based model (ResNet architecture)** is trained on labeled seismic image data to automate subsurface interpretation, which is traditionally a manual and time-consuming task in the oil & gas domain.

---

## 🧠 Problem Statement
Manual interpretation of seismic data is complex, time-intensive, and prone to human error.  
This project aims to **leverage deep learning** to assist geophysicists by accurately classifying seismic images based on hydrocarbon presence.

---

## ⚙️ Tech Stack
- Python
- TensorFlow / Keras
- ResNet (Transfer Learning)
- NumPy
- Matplotlib

---

## 📂 Project Structure

Seismic_DL_proj/
│
├── seismic.py # Model training & evaluation script
├── requirements.txt # Project dependencies
├── README.md # Project documentation
├── data/ # Seismic image dataset (ignored in repo)
│ ├── train/
│ ├── val/
│ └── test/
├── models/ # Trained model files (ignored in repo)

## 📊 Dataset
The dataset consists of labeled **seismic images** divided into:
- Training set
- Validation set
- Test set  

Each image belongs to one of the following classes:
- `with_hydrocarbon`
- `without_hydrocarbon`

> ⚠️ Dataset is not included in this repository due to size constraints.

---

## 🏗️ Model Architecture
- Pretrained **ResNet** used as the base model
- Fine-tuned using transfer learning
- Final classification layer for binary classification

---

## 🚀 How to Run the Project
1. Clone the repository:
```bash
git clone https://github.com/cheetosh/Seismic_DL_proj.git
cd Seismic_DL_proj

pip install -r requirements.txt
python seismic.py

---

## ✅ 2️⃣ `.gitignore` (COPY–PASTE THIS FILE)

```gitignore
# Ignore datasets and trained models
data/
models/

# Python
__pycache__/
*.pyc

# Virtual environments
.env
.venv
env/

# IDE files
.vscode/
.idea/
