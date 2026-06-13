from config import settings

def assess_crop_condition(predicted_moisture, projected_temperature):
    """
    Lightweight rule-based tabular crop condition classifier.
    
    Args:
        predicted_moisture (float): Predicted soil moisture for the next day.
        projected_temperature (float): Expected or current temperature.
        
    Returns:
        tuple: (Crop Class String, Explanation String)
    """
    # Define optimal moisture and critical moisture points
    optimal_moisture = settings.MOISTURE_THRESHOLD + 10.0 # Just above the irrigation threshold
    critical_moisture = settings.MOISTURE_THRESHOLD - 5.0 # Well below the threshold
    
    # Heat stress threshold
    heat_stress_temp = 35.0
    
    if predicted_moisture >= optimal_moisture:
        if projected_temperature > heat_stress_temp:
            return "WARNING", "Soil moisture is optimal, but high temperatures may induce heat stress."
        return "SAFE", "Soil moisture is optimal and temperature is within safe limits."
        
    elif predicted_moisture >= critical_moisture:
        if projected_temperature > heat_stress_temp:
             return "STRESS", "Soil moisture is below optimal range and high temperatures are inducing severe stress."
        return "WARNING", "Soil moisture is below optimal range but not critically low. Moderate deficit."
        
    else:
        return "STRESS", "Severe soil moisture deficit detected. Immediate action required to prevent crop damage."
