# American Sign Language with Deep Learning
# TEAM_MEMBERS
# ANGEL SARAH JOSEPHINE B- 36822101
# INDHU MATHI K-	36822104
# RITHISH R-	36822112
# SUNIL KUMAR M-	36822114
# SYED ALJIBRE A-	36822115
# TEST CASE
import cv2
import numpy as np
from keras.models import load_model

# Load your trained model
model = load_model("C:\\Users\\rithi\\Desktop\\ASL DL\\Model\\asl_model.h5")  # Replace with the path to your trained model

# Define the labels corresponding to the classes
class_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q",
                "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Function to perform ASL hand sign recognition
def recognize_asl_hand_sign(image_path):

    # Load the image
    frame = cv2.imread(image_path)

    # Preprocess the frame (resize to the same dimensions as your training data)
    frame = cv2.resize(frame, (100, 100))

    # Expand dimensions to match the model's input shape
    input_data = np.expand_dims(frame, axis=0)

    # Normalize the input data
    input_data = input_data.astype('float32') / 255

    # Make predictions
    predictions = model.predict(input_data)
    predicted_class = class_labels[np.argmax(predictions)]
    return predicted_class

image_path = ("C:\\Users\\rithi\\Desktop\\ASL DL\\Dataset\\ASL Alphabet\\Source\\D\\F109.jpg")  # Replace with the actual path to your image

# Perform ASL hand sign recognition
predicted_class = recognize_asl_hand_sign(image_path)

# Display the predicted class label
print(f'Predicted ASL Hand Sign: {predicted_class}')