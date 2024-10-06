import os
import numpy as np
import streamlit as st
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import openai  # Import the OpenAI library

# Parameters
image_size = (224, 224)
model_file = r'C:\Users\me101\Desktop\test\2024_Owl_Hacks\brain_tumor_classifier.h5'

# OpenAI API setup - load the API key securely from an environment variable
openai.api_key = "" 
if not openai.api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    exit(1)

# Load the model
if os.path.exists(model_file):
    model = load_model(model_file)
    st.success("Model loaded successfully.")
else:
    st.error("Model file does not exist. Please check the path.")
    exit(1)

# Function to predict the class of a new image
def predict_image(image_path, model):
    try:
        img = load_img(image_path, target_size=image_size)
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        return class_index
    except Exception as e:
        st.error(f"Error predicting image: {e}")
        return None

# Define class labels based on your training data structure
classes = ["giloma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# Function to get a response from the OpenAI API
def get_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Choose the appropriate model
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Error: Unable to get a response. {str(e)}")  # Display error in Streamlit
        return "Error in API call."

# Streamlit app layout
st.title("MRI Image Classifier and Tumor Information Chatbot")
st.write("Upload an MRI image to classify it.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file
    image_path = "temp_image." + uploaded_file.name.split('.')[-1]
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process the image and get prediction
    class_index = predict_image(image_path, model)

    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)  # Clean up the temporary file

    # Display the result
    if class_index is not None:
        predicted_tumor = classes[class_index]
        st.write("**Prediction Result:**")
        st.write(f"Predicted Class: {predicted_tumor.replace('_', ' ').title()}")
    else:
        st.write("Could not make a prediction.")
        predicted_tumor = None
else:
    predicted_tumor = None

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Button to learn more about the predicted tumor
if predicted_tumor:
    if st.button(f"Learn more about {predicted_tumor.replace('_', ' ').title()}"):
        prompt = f"Tell me about {predicted_tumor.replace('_', ' ')}. Include brief information about symptoms, diagnosis, and treatment options."
        bot_response = get_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

# Display chat history
if st.session_state.messages:
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            st.text_area("Bot:", message["content"], height=50, disabled=True)
