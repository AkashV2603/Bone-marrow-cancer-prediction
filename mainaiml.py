from src.data_processing.data_loader import DataLoader
from src.data_processing.preprocessing import Preprocessor
from src.models.neural_network import NeuralNetworkModel
from src.models.random_forest import RandomForestModel
from src.models.hybrid_model import HybridModel
from src.utils.visualization import plot_training_history, plot_confusion_matrix
from src.config import *
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf
import numpy as np
import os

def main():

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"Found {len(physical_devices)} GPU(s) available.")
        # Enable memory growth for all GPUs
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        print("No GPU detected, using CPU.")

    # Initialize data loader and preprocessor
    print("Loading data...")
    data_loader = DataLoader(DATA_DIR)
    
    # Get data generators with test set
    train_dataset, val_dataset, test_dataset = data_loader.get_datasets(
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Initialize models
    print("\nInitializing models...")
    nn_model = NeuralNetworkModel(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
    rf_model = RandomForestModel()
    hybrid_model = HybridModel(nn_model, rf_model)
    
    # Define model save path
    model_save_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'new_model.h5')

    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            update_freq='epoch'
        )
    ]
    
    # Train the model with generator
    print("\nTraining model...")
    with tf.device('/GPU:0' if len(physical_devices) > 0 else '/CPU:0'):  # Use GPU if available
        history = hybrid_model.fit_generator(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=callbacks
        )
    
    # Evaluate the model
    print("\nEvaluating model...")
    results = hybrid_model.evaluate(test_dataset)
    
    # Print results
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    # Generate predictions for confusion matrix
    y_pred = hybrid_model.predict(test_dataset)
    y_true = np.concatenate([y for x, y in test_dataset])
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, data_loader.classes)

if __name__ == "__main__":
    main()