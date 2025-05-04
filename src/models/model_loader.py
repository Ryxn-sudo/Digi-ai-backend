import os
import numpy as np
import tensorflow as tf
from src.config import MODEL_CONFIG

def load_model():
    """
    Load the appropriate model (multi-digit or single-digit as fallback)
    Returns:
        tuple: (model, is_multi_digit) where model is the loaded model and 
               is_multi_digit is a boolean indicating if it's a multi-digit model
    """
    print("Loading digit recognition model...")
    
    # Try loading multi-digit model first
    model_path = MODEL_CONFIG['MULTI_DIGIT_MODEL_PATH']
    fallback_model_path = MODEL_CONFIG['SINGLE_DIGIT_MODEL_PATH']
    
    print(f"Looking for multi-digit model at: {model_path}")
    
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print("Multi-digit model loaded successfully!")
            is_multi_digit = True
        elif os.path.exists(fallback_model_path):
            model = tf.keras.models.load_model(fallback_model_path)
            print("Single-digit model loaded as fallback!")
            is_multi_digit = False
        else:
            raise FileNotFoundError(f"No model found at either {model_path} or {fallback_model_path}")
        
        # Test prediction to ensure model works
        input_shape = MODEL_CONFIG['MULTI_DIGIT_INPUT_SHAPE'] if is_multi_digit else MODEL_CONFIG['SINGLE_DIGIT_INPUT_SHAPE']
        test_input = np.zeros((1,) + input_shape)
        test_pred = model.predict(test_input, verbose=0)
        print("Model test prediction successful!")
        return model, is_multi_digit
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, False
