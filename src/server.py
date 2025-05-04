from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os

from src.config import ensure_directories_exist
from src.models.model_loader import load_model
from src.services.image_processing import preprocess_base64_image
from src.services.prediction import predict_with_digit_length_detection
from src.services.contribution import load_contribution_status, save_contribution

def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    CORS(app)
    
    # Ensure required directories exist
    ensure_directories_exist()
    
    # Load model on startup
    model, is_multi_digit = load_model()
    
    @app.route('/', methods=['GET'])
    def home():
        return jsonify({
            'status': 'Server is running',
            'model_loaded': model is not None,
            'model_type': 'multi-digit' if is_multi_digit else 'single-digit'
        })
    
    @app.route('/predict', methods=['POST'])
    def predict():
        if model is None:
            error_msg = "Model not loaded. Please check server logs."
            print(error_msg)
            return jsonify({'error': error_msg}), 500
        
        try:
            # Get the image from the request
            data = request.json
            if not data or 'image' not in data:
                return jsonify({'error': 'No image data received'}), 400
                
            image_data = data['image']
            debug_mode = data.get('debug', False)
            
            # Preprocess the image
            img, digit_count_estimate, preprocessed_image = preprocess_base64_image(image_data, is_multi_digit)
            
            # Debug print
            print(f"Preprocessed image shape: {img.shape}, estimated digits: {digit_count_estimate}")
            
            # Reshape for prediction
            img = np.expand_dims(img, axis=0)
            
            # Get prediction
            if is_multi_digit:
                # For multi-digit model
                prediction, digit_type, confidence = predict_with_digit_length_detection(model, img, digit_count_estimate)
                print(f"Prediction successful: {digit_type} number={prediction}, confidence={confidence:.1f}%")
                
                response_data = {
                    'prediction': prediction,
                    'digit_type': digit_type,
                    'confidence': confidence,
                    'estimated_digit_count': digit_count_estimate,
                }
                
                # Add preprocessed image for debugging if requested
                if debug_mode:
                    response_data['preprocessed_image'] = preprocessed_image
                    
                return jsonify(response_data)
            else:
                # For single-digit model (original behavior)
                prediction = model.predict(img, verbose=0)
                predicted_digit = int(np.argmax(prediction))
                confidence = float(np.max(prediction) * 100)
                
                print(f"Prediction successful: digit={predicted_digit}, confidence={confidence:.1f}%")
                
                return jsonify({
                    'prediction': predicted_digit,
                    'confidence': confidence
                })
        
        except Exception as e:
            error_msg = f"Prediction error: {str(e)}"
            print(error_msg)
            return jsonify({'error': error_msg}), 500
            
    @app.route('/contribute', methods=['POST'])
    def contribute():
        try:
            # Get the image and digit from the request
            data = request.json
            if not data or 'image' not in data or 'digit' not in data:
                return jsonify({'error': 'Missing image data or digit label'}), 400
                
            image_data = data['image']
            digit = int(data['digit'])
            
            if digit < 0 or digit > 9:
                return jsonify({'error': 'Digit must be between 0 and 9'}), 400
            
            # Save the contribution
            contribution_id, status = save_contribution(image_data, digit)
            
            # Calculate remaining contributions needed for retraining
            next_target = status.get('next_retrain_target', 50)
            unprocessed = status.get('unprocessed_contributions', 0)
            remaining = next_target - unprocessed
            remaining = max(0, remaining)
            
            print(f"Received contribution for digit {digit}, saved as {contribution_id}")
            print(f"Contributions so far: {unprocessed}/{next_target} needed for retraining")
            
            return jsonify({
                'success': True,
                'message': 'Thank you for your contribution!',
                'contribution_id': contribution_id,
                'remaining_for_retraining': remaining,
                'current_count': unprocessed
            })
        
        except Exception as e:
            error_msg = f"Contribution error: {str(e)}"
            print(error_msg)
            return jsonify({'error': error_msg}), 500
    
    @app.route('/contribution-status', methods=['GET'])
    def contribution_status():
        """Get the current contribution status"""
        try:
            status = load_contribution_status()
            return jsonify(status)
        except Exception as e:
            error_msg = f"Error getting contribution status: {str(e)}"
            print(error_msg)
            return jsonify({'error': error_msg}), 500
            
    return app
