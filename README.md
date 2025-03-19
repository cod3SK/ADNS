Anomaly Detection in Network Security (ADNS)

1. Project Overview

Problem Definition & Scope
	•	Problem: Detecting malicious network activity using machine learning.
	•	Importance: Traditional security systems struggle with evolving cyber threats.
	•	Scope:
	•	Focus on ML-based anomaly detection.
	•	Limitations: Availability of labeled data, real-time vs. offline analysis.

⸻

2. Project Breakdown

Subtask 1: Data Collection & Preprocessing
	•	Goal: Gather and preprocess network traffic data for training ML models.
	•	Datasets Used:
	•	CICIDS2017 – Contains labeled network traffic for intrusion detection.
	•	CSE-CIC-IDS2018 – Simulates realistic attack scenarios.
	•	Preprocessing Steps:
	•	Feature extraction (e.g., flow duration, packet count).
	•	Data normalization and handling missing values.

Subtask 2: Model Selection & Evaluation
	•	Goal: Compare different ML models for anomaly detection.
	•	Techniques Used:
	•	Feature Selection: Identify key network traffic attributes.
	•	Support Vector Machine (SVM): Classifies traffic based on decision boundaries.
	•	Stochastic Gradient Descent (SGD): Optimizes classification for large datasets.
	•	Evaluation Metrics: Precision, Recall, F1-score, False Positive Rate.

Subtask 3: Deployment & Practical Use
	•	Goal: Implement real-time threat detection.
	•	Deployment Tools: Docker, cloud-based deployment, real-time alerting systems.

⸻

3. Research on Existing Technologies

Paper 1: “Network Traffic Anomaly Detection”
	•	Goal: Detect unusual network activity without predefined attack signatures.
	•	Techniques Used:
	•	PCA-based Detection: Identifies deviations from normal traffic patterns.
	•	Sketch-based Detection: Uses probabilistic data structures to detect anomalies.
	•	Signal Analysis: Applies Fourier Transform and Wavelet Analysis to detect irregular traffic.
	•	Paper Link: Network Traffic Anomaly Detection

Paper 2: “Anomaly Detection in NetFlow Network Traffic”
	•	Goal: Use supervised ML to classify normal vs. malicious traffic.
	•	Techniques Used:
	•	Feature Selection: Extracts key traffic attributes (packet count, duration, flow size).
	•	Support Vector Machine (SVM): Classifies network flows based on optimal boundaries.
	•	Stochastic Gradient Descent (SGD): Optimizes classification models for large-scale traffic.
	•	Paper Link: Anomaly Detection in NetFlow Network Traffic

⸻

4. Reproducible Sources

Datasets:
	•	CICIDS2017 Dataset
	•	CSE-CIC-IDS2018 Dataset

Code Repositories:
	•	Network Traffic Analysis and Anomaly Detection
	•	netml - ML for Network Traffic
	•	Anomaly Detection Using Isolation Forest & Deep Learning

Pretrained Models:
	•	TensorFlow Model Zoo
	•	PyTorch Model Hub

⸻

5. Evaluation of Existing Solutions
	•	Limitations of Current Approaches:
	•	High false positive rates.
	•	Limited labeled datasets for supervised learning.
	•	Difficulty in detecting zero-day attacks.
	•	Proposed Improvements:
	•	Hybrid models: Combine unsupervised (autoencoders) and supervised learning.
	•	Self-supervised learning: Train models without labeled data.
	•	Adaptive models: Continuously learn from new traffic patterns.
