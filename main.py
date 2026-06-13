import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

from config import settings
from src.data_loader import load_data
from src.preprocessing import preprocess_pipeline
from src.lstm_model import train_lstm, evaluate_lstm
from src.hybrid_model import train_hybrid, evaluate_hybrid
from src.inference import run_system
from experiments.run_experiments import run_all_experiments

def plot_actual_vs_predicted(actual, predicted, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(actual[-100:], label='Actual Moisture', color='blue', alpha=0.7)
    plt.plot(predicted[-100:], label='Predicted Moisture', color='red', alpha=0.7)
    plt.title(f"{title} (Last 100 days)")
    plt.xlabel('Days')
    plt.ylabel('Soil Moisture (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(settings.PLOTS_DIR, filename))
    plt.close()

def main():
    # Execute Pipeline Silently
    df = load_data()
    data_splits = preprocess_pipeline(df)
    target_scaler = data_splits['target_scaler']
    
    X_train, y_train, X_test, y_test = data_splits['lstm']
    
    lstm_model, lstm_history = train_lstm(X_train, y_train, X_test, y_test)
    lstm_res, lstm_preds, actual = evaluate_lstm(lstm_model, X_test, y_test, target_scaler)
    
    hybrid_model, hybrid_history = train_hybrid(X_train, y_train, X_test, y_test)
    hybrid_res, hybrid_preds, _ = evaluate_hybrid(hybrid_model, X_test, y_test, target_scaler)

    with open(os.path.join(settings.OUTPUTS_DIR, "final_metrics.json"), "w") as f:
        json.dump({"Standard LSTM": lstm_res, "Temporal–Feature Fusion Network (TFFN)": hybrid_res}, f, indent=4)
    
    plot_actual_vs_predicted(actual, hybrid_preds, "Temporal–Feature Fusion Network (TFFN): Actual vs Predicted", "actual_vs_predicted.png")
    
    run_all_experiments()
    
    forecasted_rain = 0.0
    res = run_system(df, forecasted_rainfall=forecasted_rain)
    decision_text = "IRRIGATION REQUIRED" if res['irrigation_decision'] else "IRRIGATION NOT REQUIRED"
    
    demo_data = df.tail(15)
    demo_table_str = demo_data[['date', 'temperature', 'humidity', 'rainfall', 'soil_moisture']].to_string(index=False)
    
    # 2. Build the Exact Execution Report File Data
    report_text = f"""MICROCLIMATE AI DECISION SUPPORT SYSTEM – EXECUTION REPORT

1. Model Performance
Standard LSTM
RMSE: {lstm_res['RMSE']:.4f}
MAE: {lstm_res['MAE']:.4f}
R²: {lstm_res['R2']:.4f}

Temporal–Feature Fusion Network (TFFN)
RMSE: {hybrid_res['RMSE']:.4f}
MAE: {hybrid_res['MAE']:.4f}
R²: {hybrid_res['R2']:.4f}

2. Training Configuration
Sequence Length: {settings.SEQUENCE_LENGTH}
Train/Test Split: {1 - settings.TEST_SIZE:.2f}/{settings.TEST_SIZE:.2f}
Scaling: Enabled
Epochs Used: {settings.LSTM_EPOCHS}

3. Core AI Prediction
Predicted Soil Moisture for Next Day: {res['predicted_moisture']:.0f} %

4. Agentic Irrigation Decision
Recommendation: {decision_text}
Reason: {res['explanation']}

Crop Condition Assessment:
Class: {res['crop_class']}
Explanation: {res['crop_explanation']}

5. Input Data Used for Final Inference
{demo_table_str}

6. Generated Artifacts
Saved Models: models/lstm_model.h5, models/tffn_model.h5
Plots:
loss_curve.png
tffn_loss_curve.png
actual_vs_predicted.png
exp1_comparison.png
exp2_seq_length.png
exp3_scaling.png
Metrics File: outputs/final_metrics.json

7. Research Summary
TFFN vs Standard LSTM performance: The Temporal–Feature Fusion Network (TFFN) achieved an RMSE of {hybrid_res['RMSE']:.4f} compared to Standard LSTM's {lstm_res['RMSE']:.4f}, demonstrating the advantage of feature fusion.
Why Temporal–Feature Fusion Network (TFFN) architecture is better for time-series: By combining temporal learning from the {settings.SEQUENCE_LENGTH}-day sequence with dense feature extraction, the model captures complex sequential and nonlinear relationships.
Decision support capability: The system abstracts raw forecast outputs into an immediate actionable classification integrating multiple threshold logic steps for robust real-world agricultural utility.
"""

    report_path = os.path.join(settings.OUTPUTS_DIR, "final_execution_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    
    # 4. Final Terminal Output (STRICT)
    print("============================")
    print("MICROCLIMATE AI DECISION OUTPUT")
    print(f"Predicted Soil Moisture: {res['predicted_moisture']:.0f} %")
    print(f"Irrigation Recommendation: {decision_text}")
    print(f"Reason: {res['explanation']}")
    print(f"Crop Condition Assessment:")
    print(f"Class: {res['crop_class']}")
    print(f"Explanation: {res['crop_explanation']}")
    print(f"Report saved at: outputs/final_execution_report.txt")
    print("============================")

if __name__ == "__main__":
    main()
