import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils import resample

import pandas as pd

data_path = 'Self-Tutorial/data/processed/model_inputs.csv'
df = pd.read_csv(data_path)

# Define features and target variable
X = df.drop(columns=['hospital_expire_flag'])  # Replace with your actual target column name if different
y = df['hospital_expire_flag']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize LightGBM model with class weights
model = lgb.LGBMClassifier(class_weight='balanced', random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # For ROC AUC

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))