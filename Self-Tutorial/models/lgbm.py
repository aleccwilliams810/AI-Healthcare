import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve
)

def load_data(data_path):
    df = pd.read_csv(data_path)
    
    # Attempt to convert all columns to numeric for model
    try:
        df = df.apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        print("Error converting columns to numeric:", e)
    
    return df

def train_model(X_train, y_train, param_grid):

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = lgb.LGBMClassifier(class_weight='balanced', random_state=42)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=25,
        scoring='roc_auc',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    print("Training model with RandomizedSearchCV...")

    random_search.fit(
        X_train_sub, y_train_sub,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=7,
        eval_metric='auc',
        verbose=True
    )

    return random_search

def display_metrics(y_test, y_pred, y_pred_proba):
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'ROC AUC Score': roc_auc_score(y_test, y_pred_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    print("\nModel Performance Metrics:")
    print(metrics_df)
    return metrics_df

def save_results(y_test, y_pred, y_pred_proba, metrics_df):
    # DataFrame for predictions
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted_Probabilities': y_pred_proba,
        'Predicted_Labels': y_pred
    })

    results_df.to_csv('models/model_results.csv', index=False)
    metrics_df.to_csv('models/model_metrics.csv', index=False)

def display_best_params(random_search):
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    print("\nBest Model Parameters and Score:")
    print(pd.DataFrame([{**best_params, 'Best ROC AUC Score': best_score}]))

def display_classification_report(y_test, y_pred):
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def main():
    data_path = 'data/processed/model_inputs.csv'
    df = load_data(data_path)

    X = df.drop(columns=['hospital_expire_flag'])
    y = df['hospital_expire_flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'num_leaves': [21, 31, 41],
        'max_depth': [-1, 10, 20],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 400],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    random_search = train_model(X_train, y_train, param_grid)
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    metrics_df = display_metrics(y_test, y_pred, y_pred_proba)
    save_results(y_test, y_pred, y_pred_proba, metrics_df)
    
    display_best_params(random_search)
    display_classification_report(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_proba)
    plot_confusion_matrix(y_test, y_pred_proba)

if __name__ == "__main__":
    main()