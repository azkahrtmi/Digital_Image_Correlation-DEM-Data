# 🏔️ Mount Merapi Deformation Detection using DIC and DEM

This application performs **Digital Image Correlation (DIC)** using **Digital Elevation Model (DEM)** data to analyze surface deformation of the Mount Merapi lava dome.  
It provides an interactive interface built with **Streamlit** and backend logic in `dic.py`.

---

## 📁 Project Structure
├── app.py # Streamlit UI for user interaction

├── dic.py # DIC logic and deformation computation

├── DEM_DATA/ # Folder for input DEM files (e.g., GeoTIFF)

└── README.md # This file

---

## ⚙️ Environment Setup

### 1. Create Virtual Environment

We recommend using **conda** for consistent dependencies:

```bash
conda create -n py2dicenv python=3.7
conda activate py2dicenv
conda install -c conda-forge pyqt
conda install -c conda-forge opencv
conda install joblib
conda install -c conda-forge numpy matplotlib==3.1.2 scipy
pip install streamlit

```
---

##  🔴 Run Program
```bash
streamlit run app.py
```
