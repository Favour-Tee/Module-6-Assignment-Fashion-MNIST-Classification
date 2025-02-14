# ----------------------------------------------
#  Module 6 Assignment: Fashion MNIST Classification
#  Task:
# 1️ Build a CNN with six layers using Keras
# 2️ Train and evaluate on Fashion MNIST dataset
# 3️ Make predictions on two images
# ----------------------------------------------

#  Step 1: Import Required Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

#  Step 2: Load and Explore the Fashion MNIST Dataset
# Load dataset from Keras datasets
fashion_mnist = keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize the data (scale pixel values between 0 and 1)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Reshape to add channel dimension (needed for CNN)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Define class names for visualization
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# Display sample image
plt.imshow(X_train[0].reshape(28, 28), cmap="gray")
plt.title(f"Label: {class_names[y_train[0]]}")
plt.show()

#  Step 3: Build the CNN Model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),  # Conv Layer 1
    layers.MaxPooling2D(2,2),  # Max Pooling 1
    layers.Conv2D(64, (3,3), activation='relu'),  # Conv Layer 2
    layers.MaxPooling2D(2,2),  # Max Pooling 2
    layers.Flatten(),  # Flatten 2D to 1D
    layers.Dense(128, activation='relu'),  # Fully Connected Layer
    layers.Dense(10, activation='softmax')  # Output Layer (10 classes)
])

#  Step 4: Compile and Train the Model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Step 5: Evaluate Model Performance
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Step 6: Make Predictions on Two Images
predictions = model.predict(X_test[:2])

# Display predictions
for i in range(2):
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Predicted: {class_names[np.argmax(predictions[i])]}")
    plt.show()


