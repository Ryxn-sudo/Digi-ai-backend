import os
import json
import uuid
import cv2
import numpy as np
from datetime import datetime
from src.config import TRAINING_DATA_CONFIG
from src.services.image_processing import preprocess_base64_image

def load_contribution_status():
    """
    Load or initialize the contribution status tracker
    
    Returns:
        dict: Current contribution status
    """
    status_file = TRAINING_DATA_CONFIG['STATUS_FILE']
    
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading contribution status: {e}")
    
    # Initialize default status
    default_status = {
        'total_contributions': 0,
        'unprocessed_contributions': 0,
        'contributions_by_digit': {str(i): 0 for i in range(10)},
        'last_contribution': None,
        'last_retrain_attempt': None,
        'last_successful_retrain': None,
        'next_retrain_target': TRAINING_DATA_CONFIG['MIN_SAMPLES_FOR_RETRAINING'],  # Target for next retraining
    }
    
    # Save the default status
    with open(status_file, 'w') as f:
        json.dump(default_status, f, indent=2)
        
    return default_status

def update_contribution_status(digit):
    """
    Update the contribution status after receiving a new contribution
    
    Args:
        digit: The digit that was contributed (0-9)
        
    Returns:
        dict: Updated status information
    """
    try:
        status = load_contribution_status()
        
        # Update counters
        status['total_contributions'] += 1
        status['unprocessed_contributions'] += 1
        status['contributions_by_digit'][str(digit)] += 1
        status['last_contribution'] = datetime.now().isoformat()
        
        # Save updated status
        with open(TRAINING_DATA_CONFIG['STATUS_FILE'], 'w') as f:
            json.dump(status, f, indent=2)
            
        return status
    except Exception as e:
        print(f"Error updating contribution status: {e}")
        return None

def save_contribution(image_data, digit):
    """
    Save a contributed digit image with metadata
    
    Args:
        image_data: Base64 encoded image data
        digit: The digit label (0-9)
        
    Returns:
        tuple: (contribution_id, status) containing the ID and updated status
    """
    # Generate unique ID for this contribution
    contribution_id = str(uuid.uuid4())
    
    # Preprocess the image as we do for predictions
    img, _, _ = preprocess_base64_image(image_data)
    
    # Save the preprocessed image
    img_path = os.path.join(TRAINING_DATA_CONFIG['IMAGES_DIR'], f"{contribution_id}.png")
    cv2.imwrite(img_path, (img * 255).astype(np.uint8))
    
    # Save metadata (the digit label and additional info)
    metadata = {
        'digit': digit,
        'timestamp': datetime.now().isoformat(),
        'processed': False  # Flag to indicate this hasn't been used for training yet
    }
    
    metadata_path = os.path.join(TRAINING_DATA_CONFIG['METADATA_DIR'], f"{contribution_id}.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    # Update contribution status
    status = update_contribution_status(digit)
    
    return contribution_id, status
