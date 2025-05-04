import numpy as np

def predict_with_digit_length_detection(model, img, digit_count_estimate=1, confidence_threshold=0.5):
    """
    Predict digits with intelligent detection of digit count
    
    Args:
        model: The loaded TensorFlow model
        img: Preprocessed image array
        digit_count_estimate: Estimated number of digits from image analysis
        confidence_threshold: Base confidence threshold
        
    Returns:
        tuple: (result, digit_type, confidence) containing the predicted number, 
               type (single/multi) and confidence level
    """
    # Get raw model predictions
    predictions = model.predict(img, verbose=0)
    
    # Get predicted digits and confidences
    predicted_digits = [int(np.argmax(pred[0])) for pred in predictions]  # Convert to int
    confidences = [float(np.max(pred[0])) for pred in predictions]  # Convert to float
    
    print(f"Raw predictions: {predicted_digits}, confidences: {[f'{c:.2f}' for c in confidences]}")
    
    # Consider the estimated digit count from image analysis
    likely_multi_digit = digit_count_estimate > 1
    
    # For single digits, be more strict about confidence
    if not likely_multi_digit:
        min_confidence = 0.6  # Higher confidence threshold for single digits
    else:
        min_confidence = 0.4  # Lower threshold for multi-digits
    
    # Strategy 1: If all predictions are the same with high confidence AND image analysis suggests single digit
    if len(set(predicted_digits[:2])) == 1 and confidences[0] > 0.85 and not likely_multi_digit:
        return str(predicted_digits[0]), "single", float(confidences[0] * 100)
    
    # Adjust confidence threshold based on digit count estimate
    adjusted_confidence = min_confidence
    if likely_multi_digit:
        # For multi-digits, use the estimated count for more accurate detection
        digit_length = digit_count_estimate
        # Get the most confident predictions for each position up to the estimated count
        result_digits = []
        for i in range(min(digit_length, 5)):
            if confidences[i] > adjusted_confidence:
                result_digits.append(str(predicted_digits[i]))
        
        result = ''.join(result_digits)
        
        # If we didn't get enough digits, try again with lower threshold
        if len(result) < digit_count_estimate - 1:
            result_digits = []
            for i in range(min(digit_length, 5)):
                if confidences[i] > adjusted_confidence * 0.7:  # Lower threshold
                    result_digits.append(str(predicted_digits[i]))
            result = ''.join(result_digits)
    else:
        # For single digit, just use the most confident prediction
        if confidences[0] > min_confidence:
            result = str(predicted_digits[0])
        else:
            # If confidence is too low, check if any prediction is very confident
            max_conf_idx = np.argmax(confidences)
            if confidences[max_conf_idx] > min_confidence:
                result = str(predicted_digits[max_conf_idx])
            else:
                result = str(predicted_digits[0])  # Default to first digit
    
    # Calculate average confidence for the chosen digits
    if len(result) > 0:
        used_confidences = [confidences[i] for i in range(len(result))]
        avg_confidence = float(sum(used_confidences) / len(used_confidences) * 100)
    else:
        result = str(predicted_digits[0])  # Fallback
        avg_confidence = float(confidences[0] * 100)
    
    result_type = "multi" if len(result) > 1 else "single"
    return result, result_type, avg_confidence
