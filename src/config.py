import os

# Base directory - use simpler path for container environment
BASE_DIR = os.environ.get('BASE_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model configuration
MODEL_CONFIG = {
    'MULTI_DIGIT_MODEL_PATH': os.path.join(BASE_DIR, 'models', 'saved_models', 'improved_multi_digit.keras'),
    'SINGLE_DIGIT_MODEL_PATH': os.path.join(BASE_DIR, 'models', 'saved_models', 'improved_single_digit.keras'),
    'MULTI_DIGIT_INPUT_SHAPE': (32, 32, 1),
    'SINGLE_DIGIT_INPUT_SHAPE': (28, 28, 1),
}

# Training data configuration
TRAINING_DATA_CONFIG = {
    'TRAINING_DATA_DIR': os.path.join(BASE_DIR, 'training_data'),
    'IMAGES_DIR': os.path.join(BASE_DIR, 'training_data', 'images'),
    'METADATA_DIR': os.path.join(BASE_DIR, 'training_data', 'metadata'),
    'STATUS_FILE': os.path.join(BASE_DIR, 'training_data', 'contribution_status.json'),
    'MIN_SAMPLES_FOR_RETRAINING': 50,
    'BALANCE_THRESHOLD': 0.3,
}

# Image processing configuration
IMAGE_PROCESSING_CONFIG = {
    'MAX_DIMENSION': 800,
    'PADDING': 10,
    'THRESHOLD_VALUE': 127,
}

# Initialize app directories
def ensure_directories_exist():
    """Ensure all required directories exist"""
    directories = [
        os.path.join(BASE_DIR, 'training_data'),
        os.path.join(BASE_DIR, 'training_data', 'images'),
        os.path.join(BASE_DIR, 'training_data', 'metadata'),
        os.path.join(BASE_DIR, 'models', 'saved_models'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensuring directory exists: {directory}")
