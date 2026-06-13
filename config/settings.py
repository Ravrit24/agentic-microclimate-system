import os

# Demonstration Mode
DEMO_MODE = True
# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Data generation/loading
DATA_FILE = os.path.join(DATA_DIR, "microclimate_data.csv")
NUM_SAMPLES = 1000  # For synthetic data generation

# Feature Definitions
FEATURES = ["temperature", "humidity", "rainfall"]
TARGET = "soil_moisture"

# Preprocessing & Modeling
SEQUENCE_LENGTH = 7
TEST_SIZE = 0.2
RANDOM_STATE = 42

# LSTM Parameters
LSTM_EPOCHS = 2 if DEMO_MODE else 10
LSTM_BATCH_SIZE = 32
LSTM_UNITS = 64
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Decision Engine Thresholds
MOISTURE_THRESHOLD = 30.0  # Percentage
RAIN_THRESHOLD = 3.0       # mm
