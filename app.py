import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model (change the path if necessary)
model = tf.keras.models.load_model('model/trained_model.h5')

def preprocess_image(image):
    """Preprocess the uploaded image for prediction."""
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((256, 256))  # Resize to the input size of the model
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

def predict(image_array):
    """Make a prediction on the preprocessed image array."""
    prediction = model.predict(image_array)
    return prediction[0][0]  # Return the prediction score

# Streamlit app layout
st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to get a prediction.")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for the model
    img_array = preprocess_image(img)

    # Make a prediction
    prediction_score = predict(img_array)

    # Display the prediction result
    if prediction_score >= 0.5:
        st.write("Prediction: **Pneumonia detected** (Confidence: {:.2f}%)".format(prediction_score * 100))
    else:
        st.write("Prediction: **No Pneumonia detected** (Confidence: {:.2f}%)".format((1 - prediction_score) * 100))
