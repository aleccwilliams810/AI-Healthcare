README: Predictive Modeling Project

Overview

This project implements a predictive modeling pipeline for binary classification using a Feedforward Neural Network (FFNN). The pipeline includes:

Data Preprocessing: Cleaning and scaling the data.

Feature Engineering: Dimensionality reduction with PCA, feature selection, and embedding.

Model Training: Fine-tuning hyperparameters using cross-validation and random search.

Final Model Training: Training the best model with optimal parameters on the processed dataset.

Evaluation: Generating metrics, confusion matrices, ROC curves, and training history for analysis.

Folder Structure

project/
├── models/
│   ├── final_model.h5
│   ├── training_history.pkl
│   ├── pca.pkl
├── results/
│   ├── best_params_cv.csv
│   ├── final_predictions.csv
│   ├── confusion_matrix.pkl
│   ├── roc_curve.pkl
├── data/
│   ├── processed/
│   │   ├── model_inputs.csv
│   │   ├── X_pca.parquet
│   │   ├── y_pca.parquet
├── scripts/
│   ├── pull_data.py
│   ├── encode_features.py
│   ├── embed_descriptions.py
│   ├── cluster_embeddings.py
│   ├── tune_and_cv.py
│   ├── train_model.py
│   ├── display_final_results.py
│   ├── main_pipeline.py
├── README.md

Installation

Clone the repository:

git clone <repository_url>
cd project

Set up a virtual environment:

python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Install additional tools (if required):

Install SciSpacy model:

pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_md-0.5.0.tar.gz

Usage

Run Full Pipeline

The pipeline can be executed by running the main script:

python scripts/main_pipeline.py

Individual Scripts

Pull Data: Download and preprocess raw data.

python scripts/pull_data.py

Encode Features: Apply encoding to categorical variables.

python scripts/encode_features.py

Embed Descriptions: Generate embeddings for text-based features.

python scripts/embed_descriptions.py

Cluster Embeddings: Cluster embeddings to group similar text features.

python scripts/cluster_embeddings.py

Tune and Cross-Validate: Perform hyperparameter tuning using random search.

python scripts/tune_and_cv.py

Train Final Model: Train the model using the best parameters from tuning.

python scripts/train_model.py

Display Results: Generate final results, including metrics and visualizations.

python scripts/display_final_results.py

Outputs

Model Artifacts

models/final_model.h5: The trained FFNN model.

models/training_history.pkl: Training history for analysis.

models/pca.pkl: PCA transformation object for future use.

Results

results/best_params_cv.csv: Best hyperparameters from cross-validation.

results/final_predictions.csv: Predictions and probabilities from the final model.

results/confusion_matrix.pkl: Serialized confusion matrix.

results/roc_curve.pkl: Serialized ROC curve data.

Processed Data

data/processed/X_pca.parquet: PCA-transformed features.

data/processed/y_pca.parquet: Target values.

Visualizations

The following visualizations are generated using display_final_results.py:

Metrics: Accuracy, precision, recall, and F1 score.

Confusion Matrix: Heatmap visualization.

ROC Curve: Visual representation of the model's performance.

Training History: Loss and accuracy trends during training.

Customization

Adjusting PCA Components: Modify the number of components in apply_pca to balance dimensionality and model performance.

Hyperparameter Tuning: Update param_grid in tune_and_cv.py to explore additional configurations.

Feature Selection: Modify initial_k_values to test different numbers of selected features during tuning.

Requirements

Python 3.8+

TensorFlow

SciPy

Pandas

NumPy

SciKit-Learn

Matplotlib

Seaborn

License

This project is licensed under the MIT License. See LICENSE for more details.

Acknowledgments

Special thanks to the teams behind TensorFlow, SciSpacy, and SciKit-Learn for providing the tools to build this pipeline.

