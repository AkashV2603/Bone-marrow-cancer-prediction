import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from src.data_processing.preprocessing import Preprocessor
from src.config import IMG_SIZE

def verify_image_path(image_path):
    """Verify if image path exists and is a valid image file"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    if not image_path.lower().endswith(valid_extensions):
        raise ValueError(f"Unsupported image format. Supported formats: {valid_extensions}")

def get_model_path():
    """Get the path to the model file by checking multiple possible locations"""
    possible_locations = [
        os.path.join(os.path.dirname(__file__), 'final_model.h5'),
        os.path.join(os.path.dirname(__file__), '..', 'final_model.h5'),
        os.path.join(os.path.dirname(__file__), 'models', 'final_model.h5'),
        os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.h5')
    ]
    
    for path in possible_locations:
        if os.path.exists(path):
            return path
            
    return None

def predict_image(image_path):
    """Predict the class of a single image"""
    # Define class names
    class_names = ['ABE', 'ART', 'BAS', 'BLA', 'EBO', 'EOS', 'FGC', 
                   'HAC', 'KSC', 'LYI', 'LYT', 'MMZ', 'MON', 'MYB', 
                   'NGB', 'NGS', 'NIF', 'OTH', 'PEB', 'PLM', 'PMO']
    
    try:
        # Verify image path
        verify_image_path(image_path)
        
        # Convert to absolute path
        image_path = os.path.abspath(image_path)
        
        # Load and preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize image
        img = cv2.resize(img, IMG_SIZE)
        
        # Preprocess image
        preprocessor = Preprocessor()
        img_processed = preprocessor.preprocess_image(img)
        
        # Expand dimensions to create batch
        img_batch = np.expand_dims(img_processed, axis=0)
        
        # Load model with improved path handling
        model_path = get_model_path()
        if model_path is None:
            raise FileNotFoundError(
                "Model file 'best_model.h5' not found. Please ensure the model file exists in one of these locations:\n" +
                "\n".join(["- " + os.path.abspath(p) for p in [
                    os.path.join(os.path.dirname(__file__), 'final_model.h5'),
                    os.path.join(os.path.dirname(__file__), '..', 'final_model.h5'),
                    os.path.join(os.path.dirname(__file__), 'models', 'final_model.h5'),
                    os.path.join(os.path.dirname(__file__), '..', 'models', 'final_model.h5')
                ]])
            )
            
        model = tf.keras.models.load_model(model_path)
        
        # Make prediction
        predictions = model.predict(img_batch, verbose=0)  # Suppress prediction progress bar
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # Get predicted class name
        predicted_class = class_names[predicted_class_idx]
        
        print(f"\nPrediction Results:")
        print(f"Input image: {image_path}")
        print(f"Class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        
        # Show top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        print("\nTop 3 Predictions:")
        for idx in top_3_idx:
            print(f"{class_names[idx]}: {predictions[0][idx]:.2%}")
            
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict bone marrow cell type from image')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--model', help='Path to the model file (optional)')
    args = parser.parse_args()
    
    if args.model and os.path.exists(args.model):
        model_path = args.model
    
    predict_image(args.image_path)