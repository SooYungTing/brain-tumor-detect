<div align="center">
  <h1>🧠 Brain Tumor MRI Classification</h1>
  <p>This project implements a deep learning pipeline to classify brain tumors from MRI scans using MobileNetV2 as the backbone. The model is trained and fine-tuned on a Kaggle dataset and deployed via a Streamlit app for interactive predictions.</p>

<div>
  <img src="https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/-Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras" />
  <img src="https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/-scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn" />
  <img src="https://img.shields.io/badge/-Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle" />
</div>

</div>

## 📂 Project Structure

```
    brain-tumor-classifier
    ├── .gitignore
    ├── check_labels.py      # Check label of the data
    ├── requirements.txt     # Python libraries to download
    ├── tumor.py             # Model training & evaluation script
    ├── streamlit_app.py     # Streamlit web app for predictions
    ├── brain_tumor.h5       # Trained model (generated after training)
    ├── README.md            # Project documentation

```

## 📊 Dataset

The dataset is downloaded directly from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) using **kagglehub**.

<br/>

It contains MRI scans divided into **4 classes**:
<br/>

- glioma
- meningioma
- pituitary
- notumor

## 🌟 Features

- Transfer Learning with MobileNetV2
- Two-stage training (frozen backbone $\rightarrow$ fine-tuning deeper layers)
- Data augmentation for better generalisation
- Class imbalance handling using class weights
- EarlyStopping + ReduceLROnPlateau callbacks
- Evaluation with confusion matrix
- Streamlit app for real-time predictions

## 🚀 Installation

1. **Navigate** to the path/location you want to **clone the repo**:

```bash
    git clone https://github.com/SooYungTing/brain-tumor-detect.git
    cd brain-tumor-detect
```

2. Create & Activate the virtual environment:

```bash
    #tumor here is the name of the virtual environment feel free to change it to your preference
    conda create --name tumor
    conda activate tumor
```

3. Install Python & dependencies:

```bash
    conda install -n tumor python=3.13
    python --version #check if python is installed

    #install dependencies
    python -m pip install -r requirements.txt
```

4. Train the model:

```bash
    python tumor.py
```

5. Run the streamlit app:

```bash
    streamlit run streamlit_app.py
```

## 🌐 Streamlit Demo

Upload an MRI image, and the app will predict the tumor type with probabilities across the four classes.

## 🛠️ Tech Stack

- Python 3.13
- Tensorflow / Kereas
- OpenCV
- scikit-learn
- Matplotlib & Seaborn
- Streamlit
