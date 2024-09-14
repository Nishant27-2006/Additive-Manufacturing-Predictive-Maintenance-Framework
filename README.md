# Additive-Manufacturing-Predictive-Maintenance-Framework
This repository contains the implementation of a dual-sensor system combining Acoustic Emission (AE) sensors and infrared thermography for predictive maintenance in metal-based Additive Manufacturing (AM) processes. The framework leverages machine learning models to detect and prevent defects in real-time, ensuring the reliability of mission-critical components, particularly in aerospace manufacturing.

Features
Dual-Sensor System: Integration of AE sensors for detecting internal microstructural changes and infrared thermography for external temperature anomalies.
Real-Time Monitoring: Continuous monitoring of the AM process to detect defects during the build process, improving process consistency.
Machine Learning Integration: Advanced data analysis framework using Random Forest, Logistic Regression, and Convolutional Neural Networks (CNNs) for defect prediction.
Predictive Maintenance: Enables early detection of defects and provides maintenance recommendations based on data from in-situ sensors.
Scalability: Framework adaptable for other industries such as automotive and medical device manufacturing.
Technologies Used
Acoustic Emission Sensors: Used to detect internal defects and structural changes during the AM process.
Infrared Thermography: Monitors external temperature fluctuations to identify surface-level anomalies.
Machine Learning Models:
Random Forest
Logistic Regression
Convolutional Neural Networks (CNNs)
Data Processing:
Continuous Wavelet Transform (CWT)
Principal Component Analysis (PCA)
K-means and DBSCAN clustering
Programming Languages:
Python
Libraries:
Scikit-learn
TensorFlow/PyTorch (for CNN implementation)
OpenCV (for image processing)
Numpy, Pandas (for data handling)
Getting Started
Prerequisites
Python 3.x
Libraries:
Scikit-learn
TensorFlow or PyTorch
OpenCV
Numpy
Pandas
You can install the required libraries by running:

pip install -r requirements.txt
Installation
Clone the repository:

git clone https://github.com/your-username/am-predictive-maintenance.git
cd am-predictive-maintenance
Install the required dependencies:

pip install -r requirements.txt
Usage
Sensor Data Acquisition:

Configure AE sensors and infrared thermography devices to capture real-time data from the AM process.
Use the provided Python scripts in the data_acquisition directory to collect and store data from both sensors.
Data Preprocessing:

Preprocess acoustic and thermal data using the scripts in the data_preprocessing directory.
Feature extraction techniques like Continuous Wavelet Transform (CWT) and image processing (OpenCV) are applied to sensor data.
Model Training:

Train machine learning models for defect detection using the data in the models directory.

Example commands for training Random Forest and Logistic Regression models:

python train_random_forest.py
python train_logistic_regression.py
Real-Time Monitoring:

Implement the real-time monitoring system using the real_time_monitoring directory.
Use the trained models to predict defects in real-time by processing incoming sensor data.
Predictive Maintenance:

The framework will provide predictive maintenance recommendations based on the model outputs.
You can view the results in the user interface or generate build quality reports for each production run.
Example
Run the following command to process sample data and predict defects:

python predict_defects.py --data_path data/sample_data.csv --model_path models/random_forest_model.pkl
Model Performance
Random Forest Model Accuracy: 85.96%
Logistic Regression Accuracy: 85.34%
F1-Score (Defective Class): 0.92
Contributing
Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit your changes (git commit -m 'Add feature').
Push to the branch (git push origin feature-branch).
Open a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

