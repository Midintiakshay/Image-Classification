import streamlit as st

st.set_page_config(layout="wide")

import numpy as np
import tensorflow as tf
from PIL import Image

# Function for MobileNetV2 ImageNet model
def mobilenetv2_imagenet():
    st.title("Image Classification with MobileNetV2")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Convert image to RGB if it has an alpha channel
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        st.image(image, caption='Uploaded Image', width=300)
        
        st.markdown("<p style='font-size:20px;'>Classifying...</p>", unsafe_allow_html=True)
        
        # Load MobileNetV2 model
        model = tf.keras.applications.MobileNetV2(weights='imagenet')
        
        # Preprocess the image
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
        
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.markdown(f"<p style='font-size:22px;'>{label}: {score * 100:.2f}%</p>", unsafe_allow_html=True)

# Function for CIFAR-10 model
def cifar10_classification():
    st.title("CIFAR-10 Image Classification")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Convert image to RGB if it has an alpha channel
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        st.image(image, caption='Uploaded Image', width=300)
        
        st.markdown("<p style='font-size:20px;'>Classifying...</p>", unsafe_allow_html=True)
        
        # Load CIFAR-10 model
        model = tf.keras.models.load_model('model111.h5')
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Preprocess the image
        img = image.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make predictions
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)
        
        st.markdown(f"<p style='font-size:22px;'>Predicted Class: {class_names[predicted_class]}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='font-size:22px;'>Confidence: {confidence * 100:.2f}%</p>", unsafe_allow_html=True)

# Main function to control the navigation
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Choose Model", ("CIFAR-10","MobileNetV2 (ImageNet)"))
    
    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()

if __name__ == "__main__":
    main()