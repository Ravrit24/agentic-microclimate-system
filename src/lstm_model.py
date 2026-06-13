import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def build_lstm_model(input_shape):
    """
    Builds the LSTM forecasting model architecture.
    """
    model = Sequential([
        LSTM(settings.LSTM_UNITS, activation='relu', input_shape=input_shape),
        Dropout(settings.DROPOUT_RATE),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def train_lstm(X_train, y_train, X_val, y_val):
    """
    Trains the LSTM model with early stopping and checkpointing.
    Saves history plot.
    """
    print("Training LSTM Model...")
    # X_train shape: (samples, time_steps, features)
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    model_path = os.path.join(settings.MODELS_DIR, "lstm_model.h5")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=settings.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        epochs=settings.LSTM_EPOCHS,
        batch_size=settings.LSTM_BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=0
    )
    
    # Save training loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(settings.PLOTS_DIR, "loss_curve.png"))
    plt.close()
    
    return model, history

def evaluate_lstm(model, X_test, y_test, target_scaler):
    """
    Evaluates the LSTM model.
    """
    print("Evaluating LSTM Model...")
    predictions_scaled = model.predict(X_test)
    
    # Inverse transform
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_inv = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
    mae = mean_absolute_error(y_test_inv, predictions_inv)
    r2 = r2_score(y_test_inv, predictions_inv)
    
    results = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2)
    }
    
    print(f"LSTM Evaluation Results: {results}")
    return results, predictions_inv, y_test_inv
