# Team-12-capstone
major project -disease detection through x ray images
This project focuses on developing a hybrid deep learning model that integrates both X-ray and CT scan data for early disease detection, particularly targeting respiratory and lung diseases. By combining the complementary strengths of X-rays (quick and low-cost imaging) and CT scans (detailed 3D visualization), the model aims to improve diagnostic accuracy. It employs advanced 2D and 3D Convolutional Neural Networks (CNNs) along with cross-modal feature fusion techniques to extract, combine, and analyze critical features from both imaging modalities. The goal is to assist healthcare professionals in making faster, more accurate diagnoses.
![image](https://github.com/user-attachments/assets/2df73c86-572d-498f-9717-01da87024f56)


The early detection of diseases is crucial for timely medical intervention and improved patient outcomes. In this project, we present a machine learning-based system for automatic disease detection from X-ray images. The integration of machine learning (ML) and deep learning (DL) techniques in medical imaging, particularly X-ray analysis, has shown significant promise in enhancing early disease detection. This presentation explores the application of convolutional neural networks (CNNs) and other ML methodologies to accurately identify various diseases from X-ray images. By leveraging large datasets and advanced algorithms, we aim to improve diagnostic accuracy and efficiency, ultimately contributing to better patient outcomes. This work emphasizes the importance of feature extraction, data preprocessing, and model validation in creating robust ML models capable of assisting healthcare professionals in clinical settings!
https://github.com/DEV-SPD/Healthcare.AI#features
https://github.com/DEV-SPD/Healthcare.AI#about-the-project
https://github.com/osu/AI-Powered-Disease-Detection-in-X-Ray-Images/tree/main?tab=readme-ov-file#features

#Table of Contents
Installation
Dataset
Model Architecture
Usage
Training and Evaluation
Results
OpenVINO
Contributing
License


System Architecture:
The system follows a multi-modal AI pipeline that integrates X-ray (2D) and CT scan (3D) data using deep learning techniques. The architecture consists of:

A. Input Layer (Data Acquisition)
X-ray Images:
2D images in DICOM, PNG, or JPEG formats.
Preprocessed for normalization and resizing (224×224 pixels).
CT Scan Images:
3D volumetric data sliced into 64 slices of 128×128 pixels.
Normalized and converted to NumPy arrays.
B. Feature Extraction Layer
X-ray Feature Extraction:
Uses a 2D CNN backbone like DenseNet121 or ResNet50.
Extracts global features such as lung structure, opacity, and abnormalities.
CT Scan Feature Extraction:
Uses a 3D CNN backbone like ResNet-3D or U-Net.
Captures depth-based features like nodule density and volume.
C. Cross-Modality Fusion Layer
Combines X-ray and CT scan features into a joint representation using:
Concatenation: Directly merging feature vectors.
Attention Mechanisms: Weighted feature selection based on importance.
Transformer-Based Fusion: Uses self-attention to learn correlations between both modalities.
D. Classification Head
Fully connected layers with:
ReLU activation for feature transformation.
Dropout layers to prevent overfitting.
Softmax activation for multi-class classification or sigmoid for multi-label classification.
E. Explainability Module
Grad-CAM for X-rays (generates heatmaps on suspected areas).
3D Grad-CAM for CT scans (volume-based highlight of affected regions).
F. Output Layer
Final classification of diseases such as:
Pneumonia, Tuberculosis, Lung Cancer
Output confidence score (e.g., Lung Cancer: 85% confidence).
Data Flow Diagram (DFD)
User uploads X-ray & CT scan.
Preprocessing pipeline normalizes the images.
Separate CNNs extract modality-specific features.
Fusion layer combines feature maps.
Classification model predicts disease & generates heatmaps.
Results displayed to user (disease, confidence score, and heatmap).
Technology Stack
Component	Technology Used
Model Development	TensorFlow, PyTorch
Preprocessing	OpenCV, NumPy, SciPy
Web Backend	Flask, FastAPI
Frontend	React.js, Streamlit
Deployment	AWS/GCP, Docker, Kubernetes
Database	PostgreSQL, MongoDB
Explainability	Grad-CAM, SHAP
2. MVP Demo (More than 40% Implementation)
To achieve Minimum Viable Product (MVP) with at least 40% completion, focus on the following:

1. Data Preprocessing Module (✅ Completed)
Implemented normalization, resizing, and augmentation for both X-ray and CT scans.
2. CNN Feature Extraction for X-rays (✅ Completed)
DenseNet121 trained on CheXpert dataset for pneumonia classification.
Achieved 85% accuracy on a test subset.
3. CNN Feature Extraction for CT Scans (✅ Ongoing)
Implementing ResNet-3D on LUNA16 dataset.
Current progress: Trained on 5,000 CT images with preliminary accuracy 82%.
4. Feature Fusion Layer (⚠️ In Progress)
Concatenation and attention-based fusion layer implemented.
Fine-tuning weights for better modality correlation.
5. Classification Model (⚠️ Initial Testing)
Initial model combines X-ray & CT scan features into a 512-dimensional latent space.
Current test accuracy: ~88% on pneumonia detection.
6. Explainability Module (🔜 Next Phase)
Grad-CAM heatmaps successfully generated for X-rays.
3D Grad-CAM for CT scans in the research phase.
7. Web App Prototype (⚠️ Basic Interface Ready)
Flask API built for image upload & prediction.
React-based UI under development.
8. Deployment (🔜 Next Phase)
Initial Docker containerization done for local testing.
Cloud deployment pending (AWS/GCP).
