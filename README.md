# Alzheimer's Prediction Tool

This project uses a neural network to predict early signs of Alzheimer's disease from structured patient data. The model aims to support early intervention by identifying individuals at higher risk with high sensitivity.

---

## Overview

Early diagnosis of Alzheimer's can significantly improve patient outcomes. This tool applies machine learning, specifically a feedforward neural network, to classify whether a patient is likely showing early signs of Alzheimer’s.

---

##  Model Architecture

Built using TensorFlow/Keras:

- Input Layer: Dense(64), ReLU, Batch Normalization, Dropout(0.5)
- Hidden Layer: Dense(32), ReLU, Dropout(0.5)
- Output Layer: Dense(1), Sigmoid (binary classification)

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
