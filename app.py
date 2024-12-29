import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Load your pretrained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("trained_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Function to process image
# Function to process image with improved accuracy
def segment_image(image, model):
    input_size = (128, 128)  # Match the input size of your model
    original_size = image.shape[:2]

    # Resize and normalize the image
    resized_image = cv2.resize(image, input_size)
    input_image = resized_image / 255.0  # Normalize to [0, 1]
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    # Predict segmentation mask
    prediction = model.predict(input_image)
    prediction = np.argmax(prediction, axis=-1)[0]  # Get class index for each pixel

    # Resize back to original size and apply post-processing
    segmented_image = cv2.resize(
        prediction.astype(np.uint8), 
        (original_size[1], original_size[0]), 
        interpolation=cv2.INTER_NEAREST
    )

    # Optional: Apply a smoothing filter for better visual output
    segmented_image = cv2.medianBlur(segmented_image, 3)

    return segmented_image


# Map classes to colors
def colorize_segmentation(segmented_image):
    colors = {
        0: [0, 0, 255],     # Buildings - Red
        1: [0, 255, 0],     # Trees - Green
        2: [255, 255, 0],   # Land - Yellow
        3: [0, 255, 255],   # Water - Cyan
    }

    colorized_image = np.zeros((*segmented_image.shape, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        colorized_image[segmented_image == class_id] = color
    return colorized_image

# Function to retrieve image details locally
def get_image_details(image):
    details = {
        "Dimensions": f"{image.width} x {image.height}",
        "Format": image.format or "Unknown",
        "Mode": image.mode
    }
    return details

# Add custom background CSS
def add_background():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(135deg, #72edf2 10%, #5151e5 100%);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_background()

# Streamlit interface
st.title("Semantic Land Segmentation")
st.write("Upload a satellite image to segment it into land, water, buildings, and trees.")

uploaded_file = st.file_uploader("Upload Satellite Image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Processing..."):
            np_image = np.array(image.convert("RGB"))  # Ensure image is in RGB format
            segmented_image = segment_image(np_image, model)
            colorized_image = colorize_segmentation(segmented_image)

        st.image(colorized_image, caption="Segmented Image", use_container_width=True)

        # Display image details
        if st.button("View Details"):
            details = get_image_details(image)
            st.write("### Image Details")
            for key, value in details.items():
                st.write(f"**{key}:** {value}")

        # Add refresh button
        if st.button("Refresh Page"):
            st.experimental_rerun()

        # Add download button for segmented image
        segmented_pil_image = Image.fromarray(colorized_image)
        segmented_pil_image.save("segmented_image.png")
        with open("segmented_image.png", "rb") as file:
            st.download_button(
                label="Download Segmented Image",
                data=file,
                file_name="segmented_image.png",
                mime="image/png"
            )

        # Add comparison feature
        st.write("### Compare Original and Segmented Images")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(colorized_image, caption="Segmented Image", use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
