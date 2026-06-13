import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def make_irrigation_decision(predicted_moisture, expected_rainfall):
    """
    Agentic Decision Engine Core Logic.
    
    Args:
        predicted_moisture (float): Predicted soil moisture for tomorrow
        expected_rainfall (float): Predicted rainfall for tomorrow
        
    Returns:
        tuple: (decision_boolean, explanation_string)
    """
    irrigation_required = False
    explanation_parts = []
    
    # Moisture check
    if predicted_moisture < settings.MOISTURE_THRESHOLD:
        irrigation_required = True
        explanation_parts.append(f"Soil moisture is predicted to drop to {predicted_moisture:.1f}%.")
    else:
        explanation_parts.append(f"Soil moisture is predicted to be sufficient at {predicted_moisture:.1f}%.")
        
    # Rain check
    if expected_rainfall > settings.RAIN_THRESHOLD:
        irrigation_required = False
        explanation_parts.append(f"Expected rainfall of {expected_rainfall:.1f}mm exceeds rain threshold.")
    else:
        explanation_parts.append(f"No significant rainfall expected ({expected_rainfall:.1f}mm).")
        
    # Final recommendation
    if irrigation_required:
        explanation_parts.append("Irrigation is recommended.")
    else:
        explanation_parts.append("Irrigation is NOT recommended.")
        
    explanation = " ".join(explanation_parts)
    return irrigation_required, explanation
