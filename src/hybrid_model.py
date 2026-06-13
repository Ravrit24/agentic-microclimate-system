import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Flatten, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def build_hybrid_model(input_shape):
    """
    Builds the Temporal–Feature Fusion Network (TFFN) forecasting model architecture.
    """
    seq_input = Input(shape=input_shape)
    
    # Branch 1: Temporal Feature Extraction
    lstm_out = LSTM(settings.LSTM_UNITS, activation='relu')(seq_input)
    lstm_out = Dropout(settings.DROPOUT_RATE)(lstm_out)
    lstm_dense = Dense(32, activation='relu')(lstm_out)
    
    # Branch 2: Feature Fusion Layer
    flat_out = Flatten()(seq_input)
    dense1 = Dense(64, activation='relu')(flat_out)
    dense2 = Dense(32, activation='relu')(dense1)
    
    # Feature Fusion
    merged = Concatenate()([lstm_dense, dense2])
    fc1 = Dense(32, activation='relu')(merged)
    output = Dense(1)(fc1)
    
    model = Model(inputs=seq_input, outputs=output)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=settings.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def train_hybrid(X_train, y_train, X_val, y_val):
    """
    Trains the Temporal–Feature Fusion Network (TFFN) with early stopping and checkpointing.
    Saves history plot.
    """
    print("Training Temporal–Feature Fusion Network (TFFN)...")
    model = build_hybrid_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    model_path = os.path.join(settings.MODELS_DIR, "tffn_model.h5")
    
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
    plt.title('Temporal–Feature Fusion Network (TFFN) Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(settings.PLOTS_DIR, "tffn_loss_curve.png"))
    plt.close()
    
    return model, history

def evaluate_hybrid(model, X_test, y_test, target_scaler):
    """
    Evaluates the Temporal–Feature Fusion Network (TFFN).
    """
    print("Evaluating Temporal–Feature Fusion Network (TFFN)...")
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
    
    print(f"Temporal–Feature Fusion Network (TFFN) Evaluation Results: {results}")
    return results, predictions_inv, y_test_inv
