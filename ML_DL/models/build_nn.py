import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterSampler, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

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


# Build FFNN model with L2 regularization
def build_ffnn(input_dim, hidden_layers=[64, 32], dropout_rate=0.2, learning_rate=0.001, l2_reg=0.01):
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for units in hidden_layers[1:]:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Train and evaluate model with cross-validation
def train_evaluate_model(X, y, hidden_layers, dropout_rate, n_splits=5, epochs=10, batch_size=32, l2_reg=0.01):

    if isinstance(y, pd.Series):
        y = y.values

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=23)
    metrics = []

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        model = build_ffnn(input_dim=X_train.shape[1], hidden_layers=hidden_layers, dropout_rate=dropout_rate, l2_reg=l2_reg)

        # Train the model with class weights
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_data=(X_val, y_val),
            class_weight=class_weight_dict
        )

        # Evaluate on validation set
        y_pred_prob = model.predict(X_val).ravel()
        y_pred = (y_pred_prob > 0.5).astype(int)

        fold_metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'roc_auc': roc_auc_score(y_val, y_pred_prob),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0)
        }
        metrics.append(fold_metrics)

    avg_metrics = {key: np.mean([m[key] for m in metrics]) for key in metrics[0]}
    return avg_metrics

# Perform random search for hyperparameter tuning
def random_search_ffnn(X, y, param_grid, n_iter=15, sample_fraction=0.2, epochs=10, batch_size=32, initial_k_values=[20, 30, 40]):
    param_combinations = list(ParameterSampler(param_grid, n_iter=n_iter, random_state=23))

    best_score = -np.inf
    best_params = None

    # Use a sample of the data for faster tuning
    X_sample, _, y_sample, _ = train_test_split(X, y, test_size=1 - sample_fraction, random_state=23)

    for params in param_combinations:
        for k in initial_k_values:
            print(f"Testing parameters: {params} with top {k} features")

            # Select features after PCA
            X_sample_selected, selected_features = select_features(X_sample, y_sample, k=k)

            # Evaluate model with cross-validation, passing l2_reg from params
            avg_metrics = train_evaluate_model(
                X_sample_selected, y_sample,
                hidden_layers=params['hidden_layers'],
                dropout_rate=params['dropout_rate'],
                n_splits=5, epochs=epochs, batch_size=batch_size, l2_reg=params['l2_reg']
            )

            print(f"ROC AUC for current config: {avg_metrics['roc_auc']:.4f}")

            if avg_metrics['roc_auc'] > best_score:
                best_score = avg_metrics['roc_auc']
                best_params = {**params, 'selected_features': selected_features.tolist(), 'num_features': k}

    print(f"Best ROC AUC: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    return best_params

# Full training on best hyperparameters
def full_training(X, y, best_params, epochs=30, batch_size=64, validation_split=0.2):
    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)

    # Select features for training and validation
    X_train_selected = X_train[:, best_params['selected_features']]
    X_val_selected = X_val[:, best_params['selected_features']]

    # Build model using parameters from best_params
    model = build_ffnn(
        input_dim=X_train_selected.shape[1],
        hidden_layers=best_params['hidden_layers'],
        dropout_rate=best_params['dropout_rate'],
        learning_rate=best_params['learning_rate'],
        l2_reg=best_params['l2_reg']
    )

    # Compute class weights for training
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    history = model.fit(
        X_train_selected, y_train,
        validation_data=(X_val_selected, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        class_weight=class_weight_dict
    )
    return model, history
