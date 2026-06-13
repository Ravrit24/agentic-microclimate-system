import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from src.data_loader import load_data

def train_crop_classifier_model(epochs=20, batch_size=32):
    print("Loading data for Crop Condition Classifier...")
    df = load_data()
    
    # Fill any missing values in features
    df = df.ffill().fillna(df.mean(numeric_only=True))
    
    # 5 features used by the classifier model input_shape=(5,)
    feature_cols = ['temperature', 'humidity', 'rainfall', 'canopy_temperature', 'soil_ec']
    target_col = 'crop_condition'
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Train/Val split (80/20) - shuffle is okay here as it is tabular classification
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=settings.RANDOM_STATE)
    
    # Fit scaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Save scaler for future reference
    scaler_path = os.path.join(settings.MODELS_DIR, "classifier_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved classifier scaler to {scaler_path}")
    
    # Build Keras model matching crop_classifier.h5 summary
    model = Sequential([
        Dense(64, activation='relu', input_shape=(5,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    model_path = os.path.join(settings.MODELS_DIR, "crop_classifier.h5")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
    ]
    
    print("Training Crop Condition Classifier model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_scaled, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"Saved trained classifier model to {model_path}")
    return model, history

if __name__ == "__main__":
    train_crop_classifier_model()
