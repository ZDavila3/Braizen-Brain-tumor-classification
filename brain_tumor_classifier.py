import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.applications import VGG16
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten

# Parameters
image_size = (224, 224)  # Resize images to this size
data_dir = r'C:\Users\zanet\OneDrive\Desktop\2024OwlHacks\mri_anomaly_app\archive\Training'
model_file = r'C:\Users\zanet\OneDrive\Desktop\2024OwlHacks\mri_anomaly_app\archive\brain_tumor_classifier.h5'  # Complete model path
epochs = 10  # Number of epochs for training
batch_size = 32  # Batch size for training

if os.path.exists(data_dir):
    print("Path is valid.")
else:
    print("Path does not exist.")
    exit(1)  # Exit if the data directory does not exist

# Prepare data
def prepare_data(data_dir):
    images = []
    labels = []
    classes = os.listdir(data_dir)  # This should give you the class names

    for label in classes:
        folder_path = os.path.join(data_dir, label)
        for image_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_file)
            img = load_img(img_path, target_size=image_size)  # Load and resize
            img = img_to_array(img) / 255.0  # Convert to array and normalize
            images.append(img)
            labels.append(classes.index(label))  # Assign a label based on folder index

    images = np.array(images)
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=len(classes))  # One-hot encode the labels
    return images, labels, classes

# Split dataset
def split_dataset(images, labels):
    return train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the model
def build_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Build the new model
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Number of classes

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
    return history

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Save the model
def save_model(model, filename=model_file):
    model.save(filename)
    print(f'Model saved to {filename}')

# Predict new image
def predict_image(model, image_path, classes):
    try:
        img = load_img(image_path, target_size=image_size)
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        return classes[class_index]
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None

# Main function
if __name__ == '__main__':
    # Check if the model exists
    if os.path.exists(model_file):
        print("Loading existing model...")
        model = load_model(model_file)
        
        # Optional: Load data if you want to evaluate
        images, labels, classes = prepare_data(data_dir)
        X_train, X_test, y_train, y_test = split_dataset(images, labels)
        
        evaluate_model(model, X_test, y_test)  # Evaluate loaded model (optional)
    else:
        print("No existing model found. Training a new model...")
        # Prepare the data
        images, labels, classes = prepare_data(data_dir)

        # Split the dataset
        X_train, X_test, y_train, y_test = split_dataset(images, labels)

        # Build the model
        model = build_model(len(classes))

        # Train the model
        train_model(model, X_train, y_train, X_test, y_test)

        # Save the model
        save_model(model)

    while True:
        user_image_path = input("Please enter the path to the scan image you want to classify (or type 'stop' to exit): ")

        if user_image_path.lower() == 'stop':
            print("Exiting...")
            break  # Exit the loop if the user types 'stop'

    # Predict the user-provided image
        result = predict_image(model, user_image_path, classes)
    
        if result is not None:
            print(f'Predicted Class: {result}')
        else:
            print("Could not make a prediction.")
