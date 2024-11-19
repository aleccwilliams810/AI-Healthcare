# **Predicting Patient Mortality with Feedforward Neural Networks (FFNN)**

## **Project Overview**
This project aims to develop a predictive modeling pipeline to assist hospitals in prioritizing duties and allocating resources effectively. Using a **Feedforward Neural Network (FFNN)**, the model predicts patient mortality during early hospital evaluation based on limited data from **Patients**, **Admissions**, and **Diagnosis** tables of the MIMIC-III DataBase. This project builds on the initial analysis and model developed in the Self-Tutorial.

### **Goals**
- Predict patient mortality using minimal available data at admission.
- Engineer features to maximize prediction accuracy.
- Lay the groundwork for deploying multiple models at different stages of a patient’s hospital stay.

---

## **Pipeline Overview**

### **1. Data Preparation**
- **Preprocessing**: Clean, impute, and scale raw hospital data.
- **Feature Engineering**:
  - **One-Hot encoding** for some categorical features
  -**Custom encoding** for other categorical features
  - **Text embeddings** for diagnosis descriptions with **Sci-Spacy**
  - **KMeans clustering** of embedded diagnostic descriptions
  - **Flattenting** of patient data

### **2. Model Development**
  - Dimensionality reduction with **PCA**.
  - Selection of important features via mutual information
- **Hyperparameter Tuning**: Optimize parameters using **cross-validation** and **random search**.
- **Final Training**: Train the model with selected features and best hyperparameters.

### **3. Evaluation**
- **Metrics**: ROC AUC, precision, recall, F1-score, and accuracy.
- **Visualizations**:
  - ROC Curve
  - Confusion Matrix
  - Training History (Loss and Accuracy trends)

---

## **Folder Structure**
project/
├── models/
│   ├── __init__.py             
│   ├── build_nn.py              # Model functions, including random search tuning, CV, and dimenisonality reduction
│   ├── final_model.h5           # Trained FFNN model
│   ├── training_history.pkl     # Training history
│   ├── pca.pkl                  # PCA object for reuse
│   ├── best_params_cv.json      # Optimal hyperparameters
│   ├── final_predictions.csv    # Model predictions and probabilities
│   ├── confusion_matrix.pkl     # Serialized confusion matrix
│   ├── roc_curve.pkl            # Serialized ROC curve data
├── data/
    will be dynamically filled during execution
├── scripts/
│   ├── __init__.py             
│   ├── data_manager.py          # Data collection and storage functions
│   ├── pat_processing.py        # Processing of Patient Table
│   ├── adm_processing.py        # Processing of Admissions Table
│   ├── diag_processing.py       # Processing of Diagnoses Table
│   ├── merge_and_process.py     # Join and post-process
│   ├── apply_embeddings.py      # Embedding functions
│   ├── apply_kmeans.py          # Clustering functions
│   ├── display_results.py       # Final model result display functions
├── testing/
│   ├── __init__.py             
│   ├── pull.py                  # Data collection test
│   ├── encode.py                # Encoding test
│   ├── embed.py                 # Embedding
│   ├── cluster.py               # Clustering test
│   ├── tune_and_cv.py           # Feature selection, hyperparam search, and CV testing
│   ├── train_model.py           # Final Model test
│   ├── results.py               # Test displaying results
