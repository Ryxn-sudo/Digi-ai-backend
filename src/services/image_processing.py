import cv2
import numpy as np
import base64
from src.config import IMAGE_PROCESSING_CONFIG

def preprocess_base64_image(base64_string, is_multi_digit=True):
    """
    Process a base64 encoded image for digit recognition
    
    Args:
        base64_string: Base64 encoded image string
        is_multi_digit: Boolean indicating if processing for multi-digit model
        
    Returns:
        tuple: (processed_image, digit_count_estimate, preprocessed_image_for_debug)
    """
    # Decode base64 image
    img_data = base64.b64decode(base64_string.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Store original image for digit count analysis
    original_img = img.copy()
    
    # Resize if too large (keeping aspect ratio)
    max_dimension = IMAGE_PROCESSING_CONFIG['MAX_DIMENSION']
    height, width = img.shape
    if height > max_dimension or width > max_dimension:
        scale = max_dimension / max(height, width)
        img = cv2.resize(img, (int(width * scale), int(height * scale)))
        original_img = img.copy()
    
    # Apply centering - find bounding box of content and center it
    threshold_value = IMAGE_PROCESSING_CONFIG['THRESHOLD_VALUE']
    if np.mean(img) > threshold_value:  # Light background
        _, binary = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    else:  # Dark background
        _, binary = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
        
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find bounding rectangle around all contours
        x_min, y_min = img.shape[1], img.shape[0]
        x_max, y_max = 0, 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # Add some padding around content
        padding = IMAGE_PROCESSING_CONFIG['PADDING']
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img.shape[1], x_max + padding)
        y_max = min(img.shape[0], y_max + padding)
        
        # Crop to content
        img = img[y_min:y_max, x_min:x_max]
    
    # Apply enhanced preprocessing
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Apply adaptive thresholding for better digit segmentation
    if np.mean(img) > threshold_value:  # Light background
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    else:  # Dark background
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Remove noise with morphological operations
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.dilate(img, kernel, iterations=1)
    
    # Store preprocessed image before final resizing (for debugging)
    preprocessed_for_display = img.copy()
    
    # Resize to appropriate dimensions (32x32 for multi-digit model, 28x28 for single-digit)
    img_size = 32 if is_multi_digit else 28
    
    # Ensure square aspect ratio with white padding to preserve digit proportions
    height, width = img.shape
    square_size = max(height, width)
    square_img = np.zeros((square_size, square_size), dtype=np.uint8)
    
    if np.mean(img) > threshold_value:  # If digits are white on black
        square_img.fill(0)  # Black background
    else:
        square_img.fill(255)  # White background
        
    # Center the image in the square
    offset_h = (square_size - height) // 2
    offset_w = (square_size - width) // 2
    square_img[offset_h:offset_h+height, offset_w:offset_w+width] = img
    
    # Resize the square image to target size
    img = cv2.resize(square_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    
    # Normalize and ensure proper contrast
    img = img.astype('float32') / 255.0
    
    # Adjust image contrast if needed
    if np.max(img) - np.min(img) < 0.8:  # Low contrast
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-6)  # Re-normalize with better contrast
    
    # Add channel dimension
    img_for_model = np.expand_dims(img, axis=-1)
    
    # Try to estimate number of digits in the image
    digit_count_estimate = estimate_digit_count(original_img)
    
    # Create a base64 representation of preprocessed image for debugging
    _, preprocessed_buffer = cv2.imencode('.png', preprocessed_for_display)
    preprocessed_base64 = base64.b64encode(preprocessed_buffer).decode('utf-8')
    preprocessed_data_url = f"data:image/png;base64,{preprocessed_base64}"
    
    return img_for_model, digit_count_estimate, preprocessed_data_url

def estimate_digit_count(img):
    """
    Estimate the number of digits in an image using image processing techniques
    
    Args:
        img: Input grayscale image
        
    Returns:
        int: Estimated digit count (1-5)
    """
    # Apply thresholding to isolate digits
    threshold_value = IMAGE_PROCESSING_CONFIG['THRESHOLD_VALUE']
    if np.mean(img) > threshold_value:  # Light background, dark digits
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:  # Dark background, light digits
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours
    significant_contours = [c for c in contours if cv2.contourArea(c) > 20]
    
    # Analyze horizontal distribution
    if len(significant_contours) <= 1:
        return 1
    
    # Check if contours are horizontally distributed (multi-digit)
    x_centers = []
    for c in significant_contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            x_centers.append(int(M["m10"] / M["m00"]))
    
    # If we have multiple horizontally distributed centers, likely multi-digit
    if len(x_centers) >= 2 and max(x_centers) - min(x_centers) > img.shape[1] // 4:
        return min(len(significant_contours), 5)  # Cap at 5 digits
    
    return 1  # Default to single digit
