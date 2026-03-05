# Covid-19 X-Ray Detection

A deep learning web app that detects Covid-19 from chest X-ray images using a custom CNN, built with TensorFlow and Streamlit.

> ⚠️ This tool is for **educational purposes only** and is not intended for medical diagnosis.

---

## Overview

This project trains a Convolutional Neural Network (CNN) from scratch to classify chest X-ray images as **COVID Positive** or **Normal**. The trained model is served through a Streamlit app where you can upload an X-ray image and get an instant prediction with a confidence score.

---

## Demo

Upload a chest X-ray (JPG/PNG) and the app will return:
- A **COVID Positive / Normal** prediction
- The model's **confidence score**

---

## Model

| Detail | Value |
|---|---|
| Architecture | Custom CNN (Sequential) |
| Input size | 224 × 224 × 3 |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Epochs trained | 6 |
| Final train accuracy | ~86% |
| Final val accuracy | ~95% |

### Architecture Summary

```
Conv2D(32) → Conv2D(64) → MaxPooling → Dropout(0.25)
→ Conv2D(128) → MaxPooling → Dropout(0.25)
→ Flatten → Dense(64) → Dropout(0.5) → Dense(1, sigmoid)
```

---

## Project Structure

```
.
├── app.py               # Streamlit app
├── Covid.ipynb          # Training notebook (run on Google Colab)
├── covid_model.keras    # Saved trained model
└── requirements.txt     # Python dependencies
```

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

Make sure `covid_model.keras` is in the same directory as `app.py`.

---

## Requirements

```
streamlit
tensorflow
Pillow
numpy
```
