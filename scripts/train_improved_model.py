import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
from scipy.io import loadmat
import matplotlib.pyplot as plt

def load_and_preprocess_mnist():
    """Load and preprocess MNIST data for better digit recognition"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape and normalize
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Convert to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Resize to 32x32 to match our model input
    x_train_resized = np.zeros((x_train.shape[0], 32, 32, 1), dtype='float32')
    x_test_resized = np.zeros((x_test.shape[0], 32, 32, 1), dtype='float32')
    
    for i in range(len(x_train)):
        x_train_resized[i, :, :, 0] = cv2.resize(x_train[i, :, :, 0], (32, 32))
    
    for i in range(len(x_test)):
        x_test_resized[i, :, :, 0] = cv2.resize(x_test[i, :, :, 0], (32, 32))
    
    return x_train_resized, y_train, x_test_resized, y_test

def load_svhn_data(data_dir='svhn_data'):
    """Load SVHN dataset for additional training data"""
    if not os.path.exists(data_dir):
        print(f"SVHN data directory '{data_dir}' not found. Skipping SVHN data.")
        return None, None, None, None
        
    train_data = loadmat(os.path.join(data_dir, 'train_32x32.mat'))
    test_data = loadmat(os.path.join(data_dir, 'test_32x32.mat'))
    
    # SVHN dataset format: X_train shape (32, 32, 3, n_samples), y_train shape (n_samples, 1)
    X_train = train_data['X'].transpose(3, 0, 1, 2)
    y_train = train_data['y'].reshape(-1)
    X_test = test_data['X'].transpose(3, 0, 1, 2)
    y_test = test_data['y'].reshape(-1)
    
    # Convert to grayscale and normalize
    X_train_gray = np.mean(X_train, axis=3).astype('float32') / 255.0
    X_test_gray = np.mean(X_test, axis=3).astype('float32') / 255.0
    
    # Reshape to add channel dimension
    X_train_gray = X_train_gray.reshape(-1, 32, 32, 1)
    X_test_gray = X_test_gray.reshape(-1, 32, 32, 1)
    
    # SVHN labels use 10 as 0, fix this
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0
    
    # One-hot encoding for labels
    y_train_onehot = to_categorical(y_train, 10)
    y_test_onehot = to_categorical(y_test, 10)
    
    return X_train_gray, y_train_onehot, X_test_gray, y_test_onehot

def create_improved_single_digit_model():
    """Create a more robust single-digit recognition model"""
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        # Third convolutional block
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        
        # Dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_improved_multi_digit_model(num_digits=5):
    """Create an improved multi-digit recognition model"""
    # Input layer
    inputs = Input(shape=(32, 32, 1))
    
    # Shared CNN feature extraction 
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)
    
    # Flatten features
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Multi-head outputs for each digit position
    outputs = []
    for i in range(num_digits):
        digit_output = Dense(128, activation='relu')(x)
        digit_output = BatchNormalization()(digit_output)
        digit_output = Dropout(0.4)(digit_output)
        digit_output = Dense(10, activation='softmax', name=f'digit_{i}')(digit_output)
        outputs.append(digit_output)
    
    # Create and compile the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Create dictionaries for loss and metrics
    losses = {f'digit_{i}': 'categorical_crossentropy' for i in range(num_digits)}
    metrics = {f'digit_{i}': 'accuracy' for i in range(num_digits)}
    loss_weights = {f'digit_{i}': 1.0 if i < 3 else 0.8 for i in range(num_digits)}
    
    model.compile(
        optimizer='adam',
        loss=losses,
        metrics=metrics,
        loss_weights=loss_weights
    )
    
    return model

def train_single_digit_model():
    """Train an improved single-digit model"""
    print("Loading MNIST data...")
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist()
    
    print("Loading SVHN data...")
    x_svhn_train, y_svhn_train, x_svhn_test, y_svhn_test = load_svhn_data()
    
    # If SVHN data is available, combine with MNIST to improve diversity
    if x_svhn_train is not None:
        # Use a subset of SVHN to prevent overwhelming MNIST patterns
        svhn_subset_size = min(20000, len(x_svhn_train))
        x_svhn_train_subset = x_svhn_train[:svhn_subset_size]
        y_svhn_train_subset = y_svhn_train[:svhn_subset_size]
        
        print(f"Combining MNIST ({len(x_train)} samples) and SVHN ({svhn_subset_size} samples)...")
        x_train = np.vstack((x_train, x_svhn_train_subset))
        y_train = np.vstack((y_train, y_svhn_train_subset))
    
    print(f"Total training samples: {len(x_train)}")
    
    # Create data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # Create the model
    model = create_improved_single_digit_model()
    print("Model summary:")
    model.summary()
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
        tf.keras.callbacks.ModelCheckpoint('improved_single_digit.keras', save_best_only=True)
    ]
    
    # Train model with data augmentation
    print("Training model with data augmentation...")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        steps_per_epoch=len(x_train) // 128,
        epochs=15,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save final model
    model.save('improved_single_digit.keras')
    print("Single-digit model saved as 'improved_single_digit.keras'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('single_digit_training_history.png')
    
    return model

def train_multi_digit_model(single_digit_model=None):
    """Train an improved multi-digit model"""
    print("Loading MNIST data...")
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist()
    
    print("Loading SVHN data...")
    x_svhn_train, y_svhn_train, x_svhn_test, y_svhn_test = load_svhn_data()
    
    # If SVHN data is available, use it instead of MNIST for multi-digit model
    if x_svhn_train is not None:
        x_train, y_train = x_svhn_train, y_svhn_train
        x_test, y_test = x_svhn_test, y_svhn_test
    
    # Create a multi-digit classifier
    num_digits = 5
    
    # Prepare multi-digit labels (duplicate single-digit labels for each position)
    y_train_multi = [y_train for _ in range(num_digits)]
    y_test_multi = [y_test for _ in range(num_digits)]
    
    # Create the model
    model = create_improved_multi_digit_model(num_digits)
    
    # If a pre-trained single-digit model is provided, use its weights for the shared CNN layers
    if single_digit_model is not None:
        print("Transferring weights from single-digit model...")
        # Extract weights from convolutional layers
        for i in range(9):  # First 9 layers are the convolutional blocks
            if i < len(single_digit_model.layers):
                try:
                    model.layers[i].set_weights(single_digit_model.layers[i].get_weights())
                    model.layers[i].trainable = False  # Freeze transferred layers
                except:
                    print(f"Could not transfer weights for layer {i}")
    
    print("Model summary:")
    model.summary()
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
        tf.keras.callbacks.ModelCheckpoint('improved_multi_digit.keras', save_best_only=True)
    ]
    
    # Create data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # Apply data augmentation manually for multi-output model
    print("Training model...")
    
    # Train with smaller batch size due to multi-output model complexity
    batch_size = 64
    history = model.fit(
        x_train, y_train_multi,
        batch_size=batch_size,
        epochs=20,
        validation_data=(x_test, y_test_multi),
        callbacks=callbacks
    )
    
    # Save final model
    model.save('improved_multi_digit.keras')
    print("Multi-digit model saved as 'improved_multi_digit.keras'")
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    for i in range(num_digits):
        plt.plot(history.history[f'digit_{i}_accuracy'], label=f'Digit {i} Accuracy')
    plt.plot(history.history['val_digit_0_accuracy'], label='Validation Accuracy', linestyle='--')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(2, 1, 2)
    for i in range(num_digits):
        plt.plot(history.history[f'digit_{i}_loss'], label=f'Digit {i} Loss')
    plt.plot(history.history['val_digit_0_loss'], label='Validation Loss', linestyle='--')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('multi_digit_training_history.png')
    
    return model

def main():
    print("Starting improved model training...")
    
    # Ask user which model to train
    print("Select which model to train:")
    print("1. Train improved single-digit model")
    print("2. Train improved multi-digit model")
    print("3. Train both models (single-digit model weights will be transferred to multi-digit model)")
    
    choice = input("Enter your choice (1, 2 or 3): ")
    
    if choice == '1' or choice == '3':
        print("\n==== Training Single-Digit Model ====")
        single_model = train_single_digit_model()
    else:
        single_model = None
    
    if choice == '2' or choice == '3':
        print("\n==== Training Multi-Digit Model ====")
        if choice == '3':
            train_multi_digit_model(single_model)
        else:
            train_multi_digit_model()
    
    print("\nTraining completed! You can now use the improved model(s) for digit recognition.")

if __name__ == "__main__":
    main()
