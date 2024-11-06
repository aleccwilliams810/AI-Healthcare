import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import csv

def load_data(data_path):
    return pd.read_csv(data_path)

def train_model(X_train, y_train, param_grid):
    model = lgb.LGBMClassifier(class_weight='balanced', random_state=42)
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        scoring='roc_auc',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    return random_search

def save_results(y_test, y_pred, y_pred_proba):
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted_Probabilities': y_pred_proba,
        'Predicted_Labels': y_pred
    })
    results_df.to_csv('models/model_results.csv', index=False)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'ROC_AUC_Score': roc_auc_score(y_test, y_pred_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }

    with open('model_metrics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        for key, value in metrics.items():
            writer.writerow([key, value])

def main():
    data_path = 'data/processed/model_inputs.csv'
    df = load_data(data_path)

    X = df.drop(columns=['hospital_expire_flag'])
    y = df['hospital_expire_flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'num_leaves': [31, 50, 70],
        'max_depth': [-1, 10, 20, 30],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 500],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    random_search = train_model(X_train, y_train, param_grid)
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    save_results(y_test, y_pred, y_pred_proba)

if __name__ == "__main__":
    main()