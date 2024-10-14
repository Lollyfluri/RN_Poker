import numpy as np
import cv2
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import keras
import tensorflow as tf

# Check versions
print("Keras version:", keras.__version__)
print("TensorFlow version:", tf.__version__)

# Load the model with custom objects if necessary
try:
    model = load_model('data_set_poker.h5')
except TypeError as e:
    print(f"Error loading model: {e}")
    # Handle the error or re-save the model with the current version of Keras/TensorFlow

# Assuming card_labels is defined somewhere in your code
card_labels = ['label1', 'label2', 'label3']  # Example labels
label_encoder = LabelEncoder()
label_encoder.fit(card_labels)

# Function to preprocess the input image
def preprocess_image(image_path):
    img_size = 224  # Image size used during training
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size, img_size))  # Resize the image to 224x224
    img = img / 255.0  # Normalize the image to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict the card from an image
def predict_card(image_path):
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make the prediction
    prediction = model.predict(processed_image)
    
    # Find the predicted class
    predicted_class_index = np.argmax(prediction)
    
    # Convert class index to label
    predicted_card = label_encoder.inverse_transform([predicted_class_index])
    return predicted_card