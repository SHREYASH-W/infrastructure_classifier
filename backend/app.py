from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
import os

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Modified model loading function in app.py
def load_model_safely(model_path):
    """
    Attempts to load the model using multiple approaches
    """
    logger = logging.getLogger(__name__)
    
    # First check if the file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    try:
        # First attempt: Try loading as h5 or keras format
        logger.info("Attempting to load model with tf.keras.models.load_model")
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        logger.warning(f"Standard loading failed: {str(e)}")
        
        try:
            # Second attempt: Try loading as SavedModel format
            saved_model_path = os.path.dirname(model_path)
            logger.info(f"Attempting to load as SavedModel from {saved_model_path}")
            return tf.saved_model.load(saved_model_path)
        except Exception as e:
            logger.warning(f"SavedModel loading failed: {str(e)}")
            
            try:
                # Third attempt: Try with custom object scope
                logger.info("Attempting to load with custom object scope")
                with tf.keras.utils.custom_object_scope({
                    'Functional': tf.keras.Model,
                    'functional': tf.keras.Model
                }):
                    return tf.keras.models.load_model(model_path, compile=False)
            except Exception as e:
                logger.error(f"All loading attempts failed: {str(e)}")
                raise

# Update the model loading code
MODEL_PATH = os.path.join(os.path.dirname(__file__), "infrastructure_model.keras")
logger.info(f"Loading model from {MODEL_PATH}")

try:
    model = load_model_safely(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def analyze_infrastructure(predictions):
    """
    Analyze predictions to determine infrastructure quality
    """
    # Convert tensor to numpy if needed
    if tf.is_tensor(predictions):
        predictions = predictions.numpy()
    
    # Convert numpy array to Python list
    predictions = predictions.tolist()
    
    # Get probabilities for each category
    bad_infrastructure_prob = predictions[0][0] + predictions[0][1]  # Class 0 + Class 1
    good_infrastructure_prob = predictions[0][2] + predictions[0][3]  # Class 2 + Class 3
    
    # Get individual class probabilities
    class_probs = predictions[0]
    specific_class = np.argmax(class_probs)
    
    return {
        'is_good': 1 if good_infrastructure_prob > bad_infrastructure_prob else 0,
        'quality_confidence': float(max(good_infrastructure_prob, bad_infrastructure_prob)),
        'specific_class': int(specific_class),
        'class_confidence': float(class_probs[specific_class]),
        'bad_infrastructure_prob': float(bad_infrastructure_prob),
        'good_infrastructure_prob': float(good_infrastructure_prob),
        'individual_probs': [float(p) for p in class_probs]
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        img_bytes = file.read()
        
        # Preprocess the image
        processed_image = preprocess_image(img_bytes)
        
        # Get predictions
        predictions = model(processed_image) if callable(model) else model.predict(processed_image)
        
        # Analyze results
        analysis = analyze_infrastructure(predictions)
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))