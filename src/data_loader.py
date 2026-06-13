import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import sys

# Ensure config module can be imported when running from any level
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def generate_synthetic_data(file_path, num_samples):
    """Generates synthetic microclimate data if no dataset is provided."""
    np.random.seed(settings.RANDOM_STATE)
    
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_samples)]
    
    # Simulate realistic microclimate ranges
    temperature = np.random.normal(loc=25, scale=5, size=num_samples) # 10 to 40 C
    humidity = np.random.normal(loc=60, scale=15, size=num_samples) # 30 to 90 %
    rainfall = np.random.exponential(scale=5, size=num_samples) # Mostly 0-10mm, some spikes
    
    df = pd.DataFrame({
        'date': dates,
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall
    })
    
    df.to_csv(file_path, index=False)
    print(f"Generated synthetic dataset at {file_path}")
    return df

def load_data():
    """
    Loads microclimate dataset, ensures time-order, and generates target if missing.
    """
    if not os.path.exists(settings.DATA_FILE):
        print(f"Dataset not found at {settings.DATA_FILE}. Generating synthetic data.")
        df = generate_synthetic_data(settings.DATA_FILE, settings.NUM_SAMPLES)
    else:
        print(f"Loading data from {settings.DATA_FILE}")
        df = pd.read_csv(settings.DATA_FILE)
        
    # Sort by date if available to ensure time-order integrity
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date').reset_index(drop=True)
    
    # Theoretical soil moisture calculation if target missing
    # soil_moisture ∝ (+ rainfall) (+ humidity) (− temperature)
    if settings.TARGET not in df.columns or df[settings.TARGET].isnull().any():
        print("Target 'soil_moisture' is missing or has NaN values. Generating/filling based on theoretical formula:")
        print("soil_moisture ∝ (+ rainfall) (+ humidity) (− temperature)")
        
        # Normalize features roughly to combine them
        # High rainfall increases moisture significantly. High humidity keeps it moist. High temp dries it.
        # Ensure values stay in a reasonable 'percentage' range 0-100
        base_moisture = 40.0
        temp_effect = (df['temperature'] - 25) * 1.5
        hum_effect = (df['humidity'] - 60) * 0.5
        rain_effect = df['rainfall'] * 2.0
        
        simulated_moisture = base_moisture + hum_effect + rain_effect - temp_effect
        simulated_moisture_clipped = np.clip(simulated_moisture, 0, 100)
        
        if settings.TARGET not in df.columns:
            df[settings.TARGET] = simulated_moisture_clipped
        else:
            df[settings.TARGET] = df[settings.TARGET].fillna(simulated_moisture_clipped)
        
    return df
