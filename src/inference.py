import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from src.decision_engine import make_irrigation_decision
from src.crop_classifier import assess_crop_condition

# Lazy load models to avoid tensorflow import overhead if not needed immediately
_lstm_model = None

def load_system_models():
    """Loads scalers and the trained LSTM model."""
    global _lstm_model
    try:
        from tensorflow.keras.models import load_model
        feature_scaler = joblib.load(os.path.join(settings.MODELS_DIR, "feature_scaler.pkl"))
        target_scaler = joblib.load(os.path.join(settings.MODELS_DIR, "target_scaler.pkl"))
        
        if _lstm_model is None:
            _lstm_model = load_model(os.path.join(settings.MODELS_DIR, "tffn_model.h5"), compile=False)
            
        return feature_scaler, target_scaler, _lstm_model
    except Exception as e:
        print(f"Error loading models or scalers: {e}. Ensure training has been run.")
        return None, None, None

def run_system(last_days_df, forecasted_rainfall=0.0):
    """
    End-to-end inference pipeline for the Agentic Decision Engine.
    
    Args:
        last_days_df (pd.DataFrame): DataFrame containing at least SEQUENCE_LENGTH days of data.
        forecasted_rainfall (float): Predicted rainfall for tomorrow.
        
    Returns:
        dict: Contains prediction, decision, and explanation.
    """
    feature_scaler, target_scaler, lstm_model = load_system_models()
    if lstm_model is None:
        return None

    # We need exactly the last `SEQUENCE_LENGTH` days
    input_data = last_days_df.tail(settings.SEQUENCE_LENGTH).copy()
    
    # Ensure correct columns
    for col in settings.FEATURES:
        if col not in input_data.columns:
            raise ValueError(f"Missing required feature column: {col}")
    
    # Scale features
    features_scaled = feature_scaler.transform(input_data[settings.FEATURES])
    
    # Reshape for LSTM: (1, seq_length, num_features)
    X_infer = features_scaled.reshape(1, settings.SEQUENCE_LENGTH, len(settings.FEATURES))
    
    # Predict Moisture (LSTM)
    pred_scaled = lstm_model.predict(X_infer, verbose=0)
    pred_moisture = target_scaler.inverse_transform(pred_scaled).flatten()[0]
    
    # Agentic Logic
    decision, explanation = make_irrigation_decision(pred_moisture, forecasted_rainfall)
    
    # Crop Condition (Rule-based)
    latest_temp = input_data.iloc[-1]['temperature']
    crop_class, crop_explanation = assess_crop_condition(pred_moisture, latest_temp)
    
    return {
        "predicted_moisture": float(pred_moisture),
        "irrigation_decision": decision,
        "explanation": explanation,
        "crop_class": crop_class,
        "crop_explanation": crop_explanation
    }

if __name__ == "__main__":
    # Test inference using dummy data
    from src.data_loader import generate_synthetic_data
    test_df = generate_synthetic_data("dummy.csv", 10)
    os.remove("dummy.csv")
    
    res = run_system(test_df)
    print("Test inference result:")
    import json
    print(json.dumps(res, indent=4))
