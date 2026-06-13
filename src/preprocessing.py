import numpy as np
import pandas as pd
import joblib
import os
import sys
import sys
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def handle_missing_values(df):
    """
    Handles missing values: forward fill first to respect time-series,
    then fallback to column mean for any remaining (e.g., at the start).
    """
    # Forward fill
    df_filled = df.ffill()
    # Fallback to mean for any remaining NaNs
    df_filled = df_filled.fillna(df_filled.mean(numeric_only=True))
    return df_filled

def scale_data(train_df, test_df, feature_cols, target_col):
    """
    MinMax scaling. Fits ONLY on train data to prevent data leakage.
    Saves scaler for inference.
    """
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_features_scaled = feature_scaler.fit_transform(train_df[feature_cols])
    test_features_scaled = feature_scaler.transform(test_df[feature_cols])
    
    train_target_scaled = target_scaler.fit_transform(train_df[[target_col]])
    test_target_scaled = target_scaler.transform(test_df[[target_col]])

    # Save scalers
    joblib.dump(feature_scaler, os.path.join(settings.MODELS_DIR, "feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(settings.MODELS_DIR, "target_scaler.pkl"))
    
    return train_features_scaled, test_features_scaled, train_target_scaled, test_target_scaled, target_scaler

def create_sequences(features_scaled, target_scaled, seq_length):
    """
    Generates (samples, sequence_length, features) for LSTM.
    Target is the moisture of the NEXT day after the sequence.
    """
    X, y = [], []
    for i in range(len(features_scaled) - seq_length):
        X.append(features_scaled[i:(i + seq_length)]) 
        y.append(target_scaled[i + seq_length])      
    return np.array(X), np.array(y)

def preprocess_pipeline(df, seq_length=None):
    """
    Full preprocessing pipeline.
    Ensures train/test split without shuffle (first 80%, last 20%).
    """
    if seq_length is None:
        seq_length = settings.SEQUENCE_LENGTH
        
    df = handle_missing_values(df)
    
    # Train/test split without shuffle (80% / 20%)
    split_idx = int(len(df) * (1 - settings.TEST_SIZE))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Scale data
    train_features_scaled, test_features_scaled, train_target_scaled, test_target_scaled, target_scaler = scale_data(
        train_df, test_df, settings.FEATURES, settings.TARGET
    )
    
    # Generate sequences
    X_train, y_train = create_sequences(train_features_scaled, train_target_scaled, seq_length)
    X_test, y_test = create_sequences(test_features_scaled, test_target_scaled, seq_length)
    
    return {
        'lstm': (X_train, y_train, X_test, y_test),
        'target_scaler': target_scaler,
        'test_df': test_df # Original test distribution, used for evaluation/plotting
    }

