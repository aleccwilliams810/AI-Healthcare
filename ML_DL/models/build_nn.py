import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterSampler, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Load and preprocess data
def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(-1)  # Replace missing values with -1, as 0 is an important value in the data
    return df

def preprocess_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Feature selection
def select_features(X, y, k):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    return X_selected, selected_features

# Dimensionality reduction using PCA
def apply_pca(X, n_components):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

# Build FFNN model
def build_ffnn(input_dim, hidden_layers=[64, 32], dropout_rate=0.2, learning_rate=0.001):
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Train and evaluate model with cross-validation
def train_evaluate_model(X, y, hidden_layers, dropout_rate, n_splits=4, epochs=10, batch_size=32):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=23)
    metrics = []

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = build_ffnn(input_dim=X_train.shape[1], hidden_layers=hidden_layers, dropout_rate=dropout_rate)

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_val, y_val))

        # Evaluate on validation set
        y_pred_prob = model.predict(X_val).ravel()
        y_pred = (y_pred_prob > 0.5).astype(int)

        fold_metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_prob),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred)
        }
        metrics.append(fold_metrics)

    avg_metrics = {key: np.mean([m[key] for m in metrics]) for key in metrics[0]}
    return avg_metrics

# Perform random search for hyperparameter tuning
def random_search_ffnn(X, y, param_grid, n_iter=10, sample_fraction=0.1, epochs=10, batch_size=32, initial_k_values=[10, 20, 30]):
    param_combinations = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=23))

    best_score = -np.inf
    best_params = None

    # Use a sample of the data for faster tuning
    X_sample, _, y_sample, _ = train_test_split(X, y, test_size=1-sample_fraction, random_state=23)

    for params in param_combinations:
        for k in initial_k_values:
            print(f"Testing parameters: {params} with top {k} features")

            # Select features after PCA
            X_sample_selected, selected_features = select_features(X_sample, y_sample, k=k)

            # Evaluate model with cross-validation
            avg_metrics = train_evaluate_model(
                X_sample_selected, y_sample,
                hidden_layers=params['hidden_layers'],
                dropout_rate=params['dropout_rate'],
                n_splits=3, epochs=epochs, batch_size=batch_size
            )

            print(f"ROC AUC for current config: {avg_metrics['roc_auc']:.4f}")

            if avg_metrics['roc_auc'] > best_score:
                best_score = avg_metrics['roc_auc']
                best_params = {**params, 'selected_features': selected_features, 'num_features': k}

    print(f"Best ROC AUC: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    return best_params

# Full training on best hyperparameters
def full_training(X, y, best_params, n_components, epochs=30, batch_size=64):
    # Apply PCA and then select top features
    X_reduced, pca = apply_pca(X, n_components=n_components)
    X_selected, _ = select_features(X_reduced, y, k=best_params['num_features'])

    model = build_ffnn(input_dim=X_selected.shape[1],
                       hidden_layers=best_params['hidden_layers'],
                       dropout_rate=best_params['dropout_rate'],
                       learning_rate=best_params['learning_rate'])

    history = model.fit(X_selected, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model, pca
