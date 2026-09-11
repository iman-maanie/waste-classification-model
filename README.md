# Waste Classification Model

A computer vision model that classifies waste into 5 categories (Glass, Metal, Paper, Plastic, Organic Material) using transfer learning, with a real-time webcam demo for live predictions.

## Overview

Built with TensorFlow/Keras using MobileNetV2 as a frozen base for transfer learning, with a custom classification head trained on a self-collected image dataset (~2,000 images across 5 classes, sourced from public GitHub datasets, targeted Google Images searches, and manual smartphone photos).

## Results

- **Test accuracy:** 90.49%
- **Validation accuracy:** peaked at 88.83%
- **ROC-AUC:** 0.9877 (macro-averaged: 0.9861) — strong discriminative ability across all classes
- Organic Material achieved perfect recall (1.00); Metal and Organic were the strongest-performing classes
- Main weakness: confusion between Glass and Plastic, due to visual similarity (transparency, reflections) under certain lighting

## Model architecture

- **Base:** MobileNetV2, pre-trained on ImageNet, frozen during initial training
- **Head:** GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0.5) → Dense(5, softmax)
- **Training:** 10 epochs, Adam optimizer (lr=0.0001), categorical crossentropy loss
- **Preprocessing:** images resized to 224×224, normalized to [0,1], augmented with random rotations/flips/zooms

## Live demo

The script includes a real-time classification mode using OpenCV — it captures webcam frames, runs them through the trained model, and displays the predicted class instantly. This demonstrates feasibility for applications like smart bins or mobile waste-sorting tools.

## Running it

```bash
pip install -r Dependencies.txt
python waste_classifier.py
```

The trained model (`my_model.keras`) is included, so it will load and go straight into evaluation/webcam mode rather than retraining from scratch. See `Dependencies.txt` for a note on a Keras version quirk when loading the saved model.

## Ethical considerations

Any live webcam use is processed in-memory for classification only — no frames or personal data are stored or transmitted. Training data diversity (backgrounds, lighting, object variety) was a deliberate focus to reduce bias across classes.
