# Anomaly Detection in Network Security (ADNS)

## 1. Project Overview

### Problem Definition & Scope
- **Problem:** Detect malicious network activity using machine learning.
- **Importance:** Cyber threats are evolving; traditional methods often miss new attack patterns.
- **Scope:**
  - Focus on ML-based anomaly detection.
  - Limitations: Availability of labeled data, challenges with real-time analysis versus offline processing.

---

## 2. Project Breakdown

### Subtask 1: Data Collection & Preprocessing
- **Goal:** Gather and preprocess network traffic data for training ML models.
- **Datasets Used:**
  - [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) – Contains labeled network traffic for intrusion detection.
  - [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) – Simulates realistic attack scenarios.
- **Preprocessing Steps:**
  - Feature extraction (e.g., flow duration, packet count).
  - Data normalization and handling missing values.

### Subtask 2: Model Selection & Evaluation
- **Goal:** Compare different ML models for anomaly detection.
- **Techniques Used:**
  - **Feature Selection:** Identify key network traffic attributes.
  - **Support Vector Machine (SVM):** Classifies network flows by finding optimal decision boundaries.
  - **Stochastic Gradient Descent (SGD):** Optimizes classification models for large-scale data.
- **Evaluation Metrics:** Precision, Recall, F1-score, False Positive Rate.

### Subtask 3: Deployment & Practical Use
- **Goal:** Implement real-time threat detection.
- **Deployment Tools:** Docker, cloud-based deployment, real-time alerting systems.

---

## 3. Research on Existing Technologies

### Paper 1: "Network Traffic Anomaly Detection"
- **Goal:** Detect unusual network activity without predefined attack signatures.
- **Techniques Used:**
  - **PCA-based Detection:** Uses Principal Component Analysis to identify deviations from normal traffic patterns.
  - **Sketch-based Detection:** Utilizes probabilistic data structures to efficiently summarize and detect anomalies.
  - **Signal Analysis:** Applies Fourier Transform and Wavelet Analysis to detect irregular traffic patterns.
- **Paper Link:** [Network Traffic Anomaly Detection](https://arxiv.org/abs/1402.0856?utm_source=chatgpt.com)

### Paper 2: "Anomaly Detection in NetFlow Network Traffic"
- **Goal:** Use supervised ML to classify normal vs. malicious network traffic.
- **Techniques Used:**
  - **Feature Selection:** Extracts key traffic attributes (packet count, duration, flow size).
  - **Support Vector Machine (SVM):** Classifies network flows by finding the optimal boundary.
  - **Stochastic Gradient Descent (SGD):** Optimizes ML models for large-scale traffic data.
- **Paper Link:** [Anomaly Detection in NetFlow Network Traffic](https://www.sciencedirect.com/science/article/pii/S2452414X23000390?utm_source=chatgpt.com)

---

## 4. Reproducible Sources

### Datasets
- [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
- [CSE-CIC-IDS2018 Dataset](https://www.unb.ca/cic/datasets/ids-2018.html)

### Code Repositories
- [Network Traffic Analysis and Anomaly Detection](https://github.com/emenmousavi/Network-Traffic-Analysis-Project)
- [netml - ML for Network Traffic](https://github.com/noise-lab/netml)
- [Anomaly Detection Using Isolation Forest & Deep Learning](https://github.com/MahmoudAbuAwd/Anomaly-Detection-in-Network-Traffic-Using-Isolation-Forest-and-Deep-Learning)

### Pretrained Models
- [TensorFlow Model Zoo](https://www.tensorflow.org/hub)
- [PyTorch Model Hub](https://pytorch.org/hub/)

---

## 5. Evaluation of Existing Solutions

- **Limitations of Current Approaches:**
  - High false positive rates.
  - Limited availability of labeled datasets.
  - Challenges in detecting zero-day attacks.
- **Proposed Improvements:**
  - **Hybrid Models:** Combine unsupervised (e.g., autoencoders) and supervised learning techniques.
  - **Self-supervised Learning:** Train models without relying solely on labeled data.
  - **Adaptive Models:** Continuously learn from new network traffic patterns.

---

## 6. Documentation & Organization

### GitHub Repository Structure
/data        # Datasets and preprocessing scripts
/models      # ML models and training scripts
/docs        # Research papers, summaries, and references
/notebooks   # Jupyter notebooks for analysis

- **Version Control:** Git is used for tracking changes and collaboration.

---

## 7. Conclusion & Next Steps

- **Summary:** ML-based anomaly detection offers enhanced cybersecurity but faces challenges such as data labeling and false positives.
- **Next Steps:**
  - Develop and test a prototype.
  - Evaluate additional datasets.
  - Explore deep learning models for improved accuracy.
