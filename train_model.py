import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import numpy as np
import os

# Define U-Net model
def unet_model(input_size=(128, 128, 3), num_classes=4):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    # Decoder
    u1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = concatenate([u1, c2])
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = concatenate([u2, c1])
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c5)
    model = Model(inputs, outputs)
    return model

# Dummy dataset generation
def generate_dummy_data(num_samples, input_size, num_classes):
    X = np.random.rand(num_samples, *input_size)  # Random images
    Y = np.random.randint(0, num_classes, (num_samples, input_size[0], input_size[1]))  # Random labels
    Y = to_categorical(Y, num_classes)  # One-hot encoding
    return X, Y

# Training script
if __name__ == "__main__":
    input_size = (128, 128, 3)
    num_classes = 4

    # Generate dummy data (replace with your dataset)
    X_train, Y_train = generate_dummy_data(100, input_size, num_classes)
    X_val, Y_val = generate_dummy_data(20, input_size, num_classes)

    # Initialize and compile model
    model = unet_model(input_size=input_size, num_classes=num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=10,
        batch_size=8
    )

    # Save trained model
    model.save("trained_model.h5")
    print("Model training complete and saved as 'trained_model.h5'.")
