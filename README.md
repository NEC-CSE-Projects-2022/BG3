
---

# BG3– **Automated Malaria Identification Using a Hybrid CNN-RNN Model on Microscopic Blood Smears**

## Team Info

* **22471A05B6 — **Hashmi Patan**
  [LinkedIn](https://www.linkedin.com/in/hashmi-patan-670865276/)
  *Work Done:* Model design, hybrid CNN–RNN architecture implementation, DenseNet121 & MobileNetV2 integration, experimentation with LSTM/GRU/BiLSTM models, result analysis, paper drafting.

* **22471A0596 —**Hemalatha Muchumari**
  [LinkedIn](https://linkedin.com/in/xxxxxxxxxx)
  *Work Done:* Dataset preprocessing, image enhancement (CLAHE, denoising, normalization), dataset splitting, validation experiments.

* **22471A05C0 —  **Anusha Pinneboina**
  [LinkedIn](https://linkedin.com/in/xxxxxxxxxx)
  *Work Done:* Literature survey, comparative analysis, evaluation metrics calculation, confusion matrix and result visualization.

---

## Abstract

Malaria is a life-threatening disease prevalent in regions with limited access to expert medical diagnosis. This project proposes an automated deep learning–based approach for malaria detection using microscopic blood smear images. The system combines powerful spatial feature extraction from pretrained Convolutional Neural Networks (CNNs) with temporal pattern learning using Recurrent Neural Networks (RNNs). CNN architectures such as DenseNet121 and MobileNetV2 are employed to extract discriminative image features, which are then classified using LSTM, GRU, and BiLSTM networks. Experimental evaluation on a publicly available NIH malaria dataset demonstrates that the DenseNet121–LSTM–GRU hybrid model achieves the best performance with **97.34% accuracy**, along with high precision, recall, and F1-score. The results confirm the effectiveness of hybrid CNN–RNN models for reliable and automated malaria diagnosis.

---

## Paper Reference (Inspiration)

👉 **[Advancing Malaria Identification From Microscopic Blood Smears Using Hybrid Deep Learning Frameworks
– Antora Dev; Mostafa M. Fouda; Leslie Kerby; Zubair Md Fadlullah ]**
Original IEEE conference paper used as the primary inspiration and reference for the project.

---

## Our Improvement Over Existing Paper

* Implemented and evaluated multiple hybrid CNN–RNN combinations systematically.
* Extensive preprocessing pipeline (CLAHE, denoising, sharpening) for improved image quality.
* Comparative analysis between DenseNet121 and MobileNetV2 feature extractors.
* Detailed evaluation using accuracy, precision, recall, F1-score, confusion matrix, and Grad-CAM explainability.
* Identified optimal hybrid architecture (**DenseNet121–LSTM–GRU**) for best performance.

---

## About the Project

**What it does:**
Automatically classifies microscopic blood smear images as **Parasitized** or **Uninfected**.

**Why it is useful:**
Reduces dependency on expert microscopists and enables fast, accurate malaria diagnosis in low-resource settings.

**Workflow:**
Blood smear image → preprocessing → CNN feature extraction → RNN sequence modeling → binary classification output.

---

## Dataset Used

👉 **[Cell Images for Detecting Malaria – NIH](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)**

**Dataset Details:**

* Total images: **27,558**
* Classes: Parasitized & Uninfected
* Balanced dataset (~13,779 images per class)
* Images captured at 100× magnification

---

## Dependencies Used

Python, TensorFlow, Keras, NumPy, OpenCV, Matplotlib, Scikit-learn

---

## EDA & Preprocessing

* Image resizing to **128 × 128**
* Noise removal using Non-Local Means denoising
* Contrast enhancement using **CLAHE**
* Image sharpening
* Pixel normalization to [0, 1]
* Class-balanced dataset splitting (80:10:10)

---

## Model Training Info

* Feature extractors: **DenseNet121, MobileNetV2**
* RNN classifiers: **LSTM, GRU, BiLSTM**
* Optimizers: Adam, SGD (with momentum)
* Callbacks: EarlyStopping, ReduceLROnPlateau
* Loss function: Binary Cross-Entropy
* Best model trained for ~17 epochs

---

## Model Testing / Evaluation

* Metrics used: Accuracy, Precision, Recall, F1-Score
* Confusion matrix analysis
* Grad-CAM visualization for explainability
* Balanced sensitivity and specificity observed

---

## Results

* **Best Model:** DenseNet121–LSTM–GRU
* **Test Accuracy:** 97.34%
* **Precision:** 0.9717
* **Recall:** 0.9724
* **F1-Score:** 0.9721
* Minimal misclassification across both classes

---

## Limitations & Future Work

* Performance depends on image quality
* Can be extended to:

  * Multi-class Plasmodium species classification
  * Transformer-based models (ViT, Swin)
  * Fine-tuning CNN layers
  * Edge/mobile deployment
  * Integration with clinical metadata

---

## Deployment Info

* Can be deployed as a **web-based diagnostic tool**
* Suitable for cloud or edge devices
* Supports real-time malaria screening in remote healthcare environments

---


