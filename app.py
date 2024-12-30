import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load your pretrained model (Fine-tuned or custom trained model)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("trained_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load user data
USER_DATA_FILE = "users_data.json"
ACTIVITY_LOG_FILE = "activity_log.json"

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def save_user_data(users_data):
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users_data, f, indent=4)

def load_activity_log():
    if os.path.exists(ACTIVITY_LOG_FILE):
        with open(ACTIVITY_LOG_FILE, "r") as f:
            return json.load(f)
    else:
        return []

def save_activity_log(log_data):
    with open(ACTIVITY_LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=4)

def authenticate_user(users_db, username, password):
    if username in users_db and users_db[username]["password"] == password:
        return True
    return False

# Process the image
def segment_image(image, model, roi=None):
    input_size = (128, 128)  # Model input size
    original_size = image.shape[:2]

    if roi:
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]  # Crop image based on ROI

    resized_image = cv2.resize(image, input_size)
    input_image = resized_image / 255.0  # Normalize
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    prediction = model.predict(input_image)
    prediction = np.argmax(prediction, axis=-1)[0]  # Get class index for each pixel

    segmented_image = cv2.resize(
        prediction.astype(np.uint8),
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST
    )
    return segmented_image

# Enhanced color mapping
def colorize_segmentation(segmented_image, selected_classes):
    # Updated color mapping with distinct and identifiable colors
    colors = {
        0: [255, 0, 0],     # Buildings - Blue
        1: [0, 255, 0],     # Trees - Green
        2: [255, 255, 0],   # Land - Yellow
        3: [0, 255, 255],   # Water - Cyan
        4: [255, 0, 255],   # Roads - Magenta (new)
        5: [0, 0, 255],     # Other - Red (new)
    }

    colorized_image = np.zeros((*segmented_image.shape, 3), dtype=np.uint8)
    for class_id, color in colors.items():
        if class_id in selected_classes:
            colorized_image[segmented_image == class_id] = color
    return colorized_image

def overlay_segmentation(image, mask, alpha=0.5):
    overlay = image.copy()
    mask_colored = colorize_segmentation(mask, selected_classes=[0, 1, 2, 3, 4, 5])  # Adjust selected classes
    return cv2.addWeighted(overlay, 1 - alpha, mask_colored, alpha, 0)

# AI-Assisted Feature Suggestions
def suggest_features(segmented_image):
    suggestions = []
    unique_classes, counts = np.unique(segmented_image, return_counts=True)
    total_pixels = segmented_image.size

    if 0 in unique_classes:  # Buildings
        building_percentage = (counts[unique_classes.tolist().index(0)] / total_pixels) * 100
        if building_percentage > 20:
            suggestions.append("High building density detected. Consider analyzing urban planning or building layouts.")

    if 1 in unique_classes:  # Trees
        tree_percentage = (counts[unique_classes.tolist().index(1)] / total_pixels) * 100
        if tree_percentage > 30:
            suggestions.append("Significant green coverage detected. You might want to study vegetation patterns.")

    if 3 in unique_classes:  # Water
        water_percentage = (counts[unique_classes.tolist().index(3)] / total_pixels) * 100
        if water_percentage > 10:
            suggestions.append("Water bodies detected. Hydrological analysis or flood risk mapping could be explored.")

    if 4 in unique_classes:  # Roads
        road_percentage = (counts[unique_classes.tolist().index(4)] / total_pixels) * 100
        if road_percentage > 5:
            suggestions.append("Roads detected. Potential for infrastructure planning and transportation analysis.")

    if len(suggestions) == 0:
        suggestions.append("No specific feature recommendations at this time.")
    return suggestions

# Segmentation Analytics
def analyze_segmentation(segmented_image):
    unique_classes, counts = np.unique(segmented_image, return_counts=True)
    total_pixels = segmented_image.size

    analytics = {
        "Class": [],
        "Pixels": [],
        "Percentage": []
    }

    class_names = {0: "Buildings", 1: "Trees", 2: "Land", 3: "Water", 4: "Roads", 5: "Other"}
    for class_id, count in zip(unique_classes, counts):
        analytics["Class"].append(class_names.get(class_id, f"Class {class_id}"))
        analytics["Pixels"].append(count)
        analytics["Percentage"].append((count / total_pixels) * 100)

    return analytics

# Model Training
def train_model(train_images, train_masks, epochs=10, batch_size=32):
    # Load a pre-trained U-Net or any segmentation model
    model = tf.keras.models.load_model("unet_pretrained_model.h5")  # Change to your pre-trained model

    model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(train_images, train_masks, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save("trained_model.h5")  # Save the model after training
    return model, history

# Image Loading Helper
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    return img_to_array(img) / 255.0  # Normalize the image

# Streamlit interface
st.title("Semantic Land Segmentation")
st.write("Upload a satellite image or video for segmentation, or train a custom model.")

# User Authentication (Login/Signup)
users_db = load_user_data()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate_user(users_db, username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Logged in successfully!")
        else:
            st.error("Invalid credentials.")

    if st.button("Sign Up"):
        if username in users_db:
            st.error("User already exists. Please log in.")
        else:
            password = st.text_input("Create Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")

            if password == confirm_password:
                users_db[username] = {"password": password, "profile": {"name": username}}
                save_user_data(users_db)
                st.success("User registered successfully! Please log in.")
            else:
                st.error("Passwords do not match.")

else:
    st.header(f"Welcome, {st.session_state.username}!")

    # Option for Dashboard or Training Mode
    mode = st.selectbox("Select Mode", ["Main Page", "Segmentation", "Training Mode"])

    if mode == "Main Page":
        # Display dashboard data
        st.subheader("User Activity Dashboard")
        
        activity_log = load_activity_log()
        activity_df = pd.DataFrame(activity_log)

        if not activity_df.empty:
            st.write("Recent Activities")
            st.dataframe(activity_df)

        st.subheader("Uploaded Media History")
        st.write("Display a history of uploaded images, videos, and other media")

        # Option to go back to the main page
        if st.button("Back to Main Page"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.experimental_rerun()

    elif mode == "Segmentation":
        uploaded_file = st.file_uploader("Upload Satellite Image or Video", type=["png", "jpg", "jpeg", "mp4", "avi"])

        if uploaded_file is not None:
            model = load_model()
            if model is not None:
                if uploaded_file.type.startswith('image'):
                    image = Image.open(uploaded_file)
                    np_image = np.array(image.convert("RGB"))
                    st.image(image, caption="Uploaded Image", use_container_width=True)

                    # ROI Selection Button
                    if st.button("Select ROI (Click and Drag)"):
                        img_bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
                        roi = cv2.selectROI("Select ROI", img_bgr)
                        cv2.destroyAllWindows()

                        if roi != (0, 0, 0, 0):  # If ROI is not empty
                            x, y, w, h = roi
                            segmented_image = segment_image(np_image, model, roi=roi)
                            colorized_image = colorize_segmentation(segmented_image, selected_classes=[0, 1, 2, 3, 4, 5])

                            st.image(colorized_image, caption="Segmented Image", use_container_width=True)

                            # AI-Assisted Feature Suggestions
                            suggestions = suggest_features(segmented_image)
                            st.subheader("AI-Assisted Feature Suggestions")
                            for suggestion in suggestions:
                                st.write(f"- {suggestion}")

                            # Segmentation Analytics
                            st.subheader("Segmentation Analytics")
                            analytics = analyze_segmentation(segmented_image)

                            fig, ax = plt.subplots()
                            ax.pie(analytics["Pixels"], labels=analytics["Class"], autopct="%1.1f%%", startangle=140)
                            ax.axis("equal")

                            # Color bar above pie chart
                            colors = {0: "Blue", 1: "Green", 2: "Yellow", 3: "Cyan", 4: "Magenta", 5: "Red"}
                            color_labels = list(colors.values())
                            color_values = list(colors.keys())
                            color_map = plt.cm.get_cmap("tab10", len(colors))

                            st.write("Color Legend:")
                            fig2, ax2 = plt.subplots(figsize=(8, 1))
                            ax2.imshow([color_values], cmap=color_map, aspect="auto")
                            ax2.set_xticks(np.arange(len(color_labels)))
                            ax2.set_xticklabels(color_labels)
                            ax2.set_yticks([])
                            st.pyplot(fig2)

                            st.pyplot(fig)

                            st.write("Detailed Analytics:")
                            for i in range(len(analytics["Class"])):
                                st.write(f"- {analytics['Class'][i]}: {analytics['Percentage'][i]:.2f}%")

                elif uploaded_file.type.startswith('video'):
                    video = cv2.VideoCapture(uploaded_file)
                    frame_rate = int(video.get(cv2.CAP_PROP_FPS))

                    while True:
                        ret, frame = video.read()
                        if not ret:
                            break
                        segmented_frame = segment_image(frame, model)
                        colorized_frame = colorize_segmentation(segmented_frame, selected_classes=[0, 1, 2, 3, 4, 5])
                        st.image(colorized_frame, caption="Segmented Frame", use_container_width=True)
                    video.release()

            # Log the activity
            activity_log = load_activity_log()
            activity_log.append({
                "username": st.session_state.username,
                "action": "Uploaded an image and performed segmentation",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_activity_log(activity_log)

    elif mode == "Training Mode":
        st.subheader("Upload Training Data")
        train_images = st.file_uploader("Upload Training Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        train_masks = st.file_uploader("Upload Training Masks", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

        if st.button("Train Model"):
            if train_images and train_masks:
                # Add code for training model
                pass
            else:
                st.error("Please upload training images and masks.")
