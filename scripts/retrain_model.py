import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import glob
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import TRAINING_DATA_CONFIG, MODEL_CONFIG

def load_contribution_data(min_samples=50, balanced_threshold=0.3):
    """
    Load contributed training data from the training_data directory
    with checks for data quality and balance
    """
    samples = []
    labels = []
    processed_count = 0
    
    # Check if training data directory exists
    metadata_dir = TRAINING_DATA_CONFIG['METADATA_DIR']
    images_dir = TRAINING_DATA_CONFIG['IMAGES_DIR']
    if not os.path.exists(metadata_dir):
        print(f"No metadata directory found at {metadata_dir}!")
        return None, None, 0, False
    
    # Get all metadata files
    metadata_files = glob.glob(os.path.join(metadata_dir, '*.json'))
    print(f"Found {len(metadata_files)} contribution files")
    
    if len(metadata_files) < min_samples:
        print(f"Not enough samples for retraining (need at least {min_samples}, found {len(metadata_files)})")
        return None, None, 0, False
    
    # Track counts for each digit
    digit_counts = Counter()
    
    for meta_file in metadata_files:
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        # Skip if already processed
        if metadata.get('processed', False):
            continue
            
        # Get image file path
        img_id = os.path.basename(meta_file).split('.')[0]
        img_path = os.path.join(images_dir, f"{img_id}.png")
        
        if not os.path.exists(img_path):
            print(f"Warning: Image file missing for {img_id}")
            continue
        
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Failed to load image {img_path}")
            continue
        
        # Check image quality
        if img.shape[0] < 20 or img.shape[1] < 20:
            print(f"Warning: Image size too small for {img_id}")
            continue
            
        if np.std(img) < 20:  # Check for almost blank images
            print(f"Warning: Image has very low contrast for {img_id}")
            continue
            
        # Normalize image
        img = img.astype('float32') / 255.0
        
        # Add channel dimension if needed
        img = np.expand_dims(img, axis=-1)
        
        # Add to datasets
        digit = metadata['digit']
        samples.append(img)
        labels.append(digit)
        digit_counts[digit] += 1
        
        # Mark as processed
        metadata['processed'] = True
        with open(meta_file, 'w') as f:
            json.dump(metadata, f)
            
        processed_count += 1
    
    if processed_count < min_samples:
        print(f"Not enough new samples for retraining (found {processed_count}, need {min_samples})")
        return None, None, 0, False
    
    # Check for class balance
    total_samples = sum(digit_counts.values())
    is_balanced = True
    
    # Print distribution of digits
    print("Digit distribution:")
    for digit in range(10):
        count = digit_counts.get(digit, 0)
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  Digit {digit}: {count} samples ({percentage:.1f}%)")
        
        # Check if any class is severely underrepresented
        if count / total_samples < balanced_threshold / 10:
            print(f"  Warning: Digit {digit} is underrepresented")
            is_balanced = False
    
    # Convert to numpy arrays
    X = np.array(samples)
    y = np.array(labels)
    
    return X, y, processed_count, is_balanced

def evaluate_model_improvement(model, model_updated, x_test, y_test):
    """
    Evaluate if the updated model actually performs better than the original
    """
    # Get original model accuracy
    y_test_cat = tf.keras.utils.to_categorical(y_test, 10)
    original_loss, original_acc = model.evaluate(x_test, y_test_cat, verbose=0)
    
    # Get updated model accuracy
    updated_loss, updated_acc = model_updated.evaluate(x_test, y_test_cat, verbose=0)
    
    print(f"Original model - Loss: {original_loss:.4f}, Accuracy: {original_acc:.4f}")
    print(f"Updated model  - Loss: {updated_loss:.4f}, Accuracy: {updated_acc:.4f}")
    
    # Check if there's significant improvement (at least 0.5%)
    if updated_acc >= original_acc + 0.005:
        print("✅ Updated model shows significant improvement!")
        return True
    elif updated_acc >= original_acc:
        print("✓ Updated model shows slight improvement")
        return True
    else:
        print("❌ Updated model performs worse than original")
        return False

def visualize_training_samples(X, y, max_samples=25):
    """Visualize some of the training samples to verify quality"""
    if X is None or len(X) == 0:
        return
        
    plt.figure(figsize=(10, 10))
    
    # Limit number of samples to display
    n_samples = min(max_samples, len(X))
    
    for i in range(n_samples):
        plt.subplot(5, 5, i+1)
        plt.imshow(X[i].squeeze(), cmap='gray')
        plt.title(f"Digit: {y[i]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('training_samples_preview.png')
    print(f"Saved preview of {n_samples} training samples to 'training_samples_preview.png'")

def create_test_set(X, y, test_size=0.2):
    """Create a test set from a portion of the training data"""
    if X is None or len(X) == 0:
        return None, None
        
    # Shuffle data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Split into train and test
    test_count = int(len(X) * test_size)
    test_indices = indices[:test_count]
    
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_test, y_test

def update_retrain_status(success, processed_count=0):
    """Update the contribution status after a retraining attempt"""
    status_file = TRAINING_DATA_CONFIG['STATUS_FILE']
    
    if not os.path.exists(status_file):
        print(f"Status file not found at {status_file}")
        return
    
    try:
        with open(status_file, 'r') as f:
            status = json.load(f)
        
        # Update status
        status['last_retrain_attempt'] = datetime.now().isoformat()
        
        if success:
            status['last_successful_retrain'] = datetime.now().isoformat()
            status['unprocessed_contributions'] -= processed_count
        
        # Save updated status
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        
        print(f"Updated contribution status: {processed_count} samples processed")
    except Exception as e:
        print(f"Error updating contribution status: {e}")

def retrain_model():
    """Retrain the model with new contributed data with quality checks"""
    print(f"Starting retraining process at {datetime.now().isoformat()}")
    
    # Step 1: Check model availability
    model_path = MODEL_CONFIG['SINGLE_DIGIT_MODEL_PATH']
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return False
        
    # Step 2: Load the current model
    try:
        model = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return False
    
    # Step 3: Load contribution data with checks
    print("Loading and checking contributed data...")
    min_samples = TRAINING_DATA_CONFIG['MIN_SAMPLES_FOR_RETRAINING']
    X, y, count, is_balanced = load_contribution_data(min_samples=min_samples)
    if X is None:
        update_retrain_status(False)
        return False
    
    print(f"Loaded {count} new training samples")
    
    # Step 4: Create a small test set from the data for validation
    print("Creating validation set from contributions...")
    X_test, y_test = create_test_set(X, y, test_size=0.2)
    
    # Step 5: Visualize some training samples for manual inspection
    visualize_training_samples(X, y)
    
    # Step 6: Prepare labels (one-hot encoding)
    y_onehot = tf.keras.utils.to_categorical(y, 10)
    
    # Step 7: Create a clone of the model for retraining
    model_updated = tf.keras.models.clone_model(model)
    model_updated.set_weights(model.get_weights())
    model_updated.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Step 8: Fine-tune model with new data
    print("Fine-tuning model...")
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2, min_lr=0.00001)
    ]
    
    # Train with more epochs if data is imbalanced
    epochs = 10 if is_balanced else 15
    
    history = model_updated.fit(
        X, y_onehot,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 9: Evaluate if the model actually improved
    print("Evaluating model improvement...")
    improved = evaluate_model_improvement(model, model_updated, X_test, y_test)
    
    if improved:
        # Step 10: Save the updated model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a backup of the current model
        backup_path = f"{model_path}_backup_{timestamp}"
        shutil.copy2(model_path, backup_path)
        print(f"Created backup of current model at {backup_path}")
        
        # Save the new model
        model_updated.save(model_path)
        print(f"Updated model saved to {model_path}")
        
        # Save training history graph
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'retraining_history_{timestamp}.png')
        print(f"Saved training history to retraining_history_{timestamp}.png")
        
        # Update the contribution status
        update_retrain_status(True, count)
        
        return True
    else:
        print("Model not updated as no improvement was detected.")
        update_retrain_status(False)
        return False

if __name__ == "__main__":
    retrain_model()
