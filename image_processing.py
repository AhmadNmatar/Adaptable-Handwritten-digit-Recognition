# the code here is meant to test a trained model... maby we need it later

from tensorflow.keras.models import load_model
import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2

# Load the saved model


model = load_model('/Users/ahmadmatar/Desktop/AIcourse/proj.test/data/full_model1.h5')
#model.summary()

# Path to the test image
image_path = '/Users/ahmadmatar/Desktop/AIcourse/proj.test/data/dig4a.png'


# Load the image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
print("shape befor processing ", image.shape)

# Apply thresholding (if necessary) to isolate the digit
_, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# Optionally, find and crop the region of interest (ROI) if the digit isn't centered
#contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#if contours:
#    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
#    image = image[y:y+h, x:x+w]

image = cv2.GaussianBlur(image, (5, 5), 0)  # Smooth the image
print("shape befor processin ", image.shape)
image = cv2.resize(image, (28, 28))  # Resize to 28x28 pixels
image = image.astype('float32') / 255.0  # Normalize pixel values (0 to 1)
image = image.reshape(1, 28, 28, 1)  # Reshape for model input
print("shape after  ", image.shape)
#image = 1.0 - image
# Predict the digit
prediction = model.predict(image)
predicted_class = np.argmax(prediction)
digit_language_mapping = {
    0: '0 (Arabic)', 1: '1 (Arabic)', 2: '2 (Arabic)', 3: '3 (Arabic)', 4: '4 (Arabic)', 5: '5 (Arabic)', 6: '6 (Arabic)', 7: '7 (Arabic)', 8: '8 (Arabic)', 9: '9 (Arabic)',
    10: '0 (Bangla)', 11: '1 (Bangla)', 12: '2 (Bangla)', 13: '3 (Bangla)', 14: '4 (Bangla)', 15: '5 (Bangla)', 16: '6 (Bangla)', 17: '7 (Bangla)', 18: '8 (Bangla)', 19: '9 (Bangla)',
    20: '0 (Devanagari)', 21: '1 (Devanagari)', 22: '2 (Devanagari)', 23: '3 (Devanagari)', 24: '4 (Devanagari)', 25: '5 (Devanagari)', 26: '6 (Devanagari)', 27: '7 (Devanagari)', 28: '8 (Devanagari)', 29: '9 (Devanagari)',
    30: '0 (English)', 31: '1 (English)', 32: '2 (English)', 33: '3 (English)', 34: '4 (English)', 35: '5 (English)', 36: '6 (English)', 37: '7 (English)', 38: '8 (English)', 39: '9 (English)',
    40: '0 (Farsi)', 41: '1 (Farsi)', 42: '2 (Farsi)', 43: '3 (Farsi)', 44: '4 (Farsi)', 45: '5 (Farsi)', 46: '6 (Farsi)', 47: '7 (Farsi)', 48: '8 (Farsi)', 49: '9 (Farsi)',
    50: '0 (Kannada)', 51: '1 (Kannada)', 52: '2 (Kannada)', 53: '3 (Kannada)', 54: '4 (Kannada)', 55: '5 (Kannada)', 56: '6 (Kannada)', 57: '7 (Kannada)', 58: '8 (Kannada)', 59: '9 (Kannada)',
    60: '0 (Swedish)', 61: '1 (Swedish)', 62: '2 (Swedish)', 63: '3 (Swedish)', 64: '4 (Swedish)', 65: '5 (Swedish)', 66: '6 (Swedish)', 67: '7 (Swedish)', 68: '8 (Swedish)', 69: '9 (Swedish)',
    70: '0 (Telugu)', 71: '1 (Telugu)', 72: '2 (Telugu)', 73: '3 (Telugu)', 74: '4 (Telugu)', 75: '5 (Telugu)', 76: '6 (Telugu)', 77: '7 (Telugu)', 78: '8 (Telugu)', 79: '9 (Telugu)',
    80: '0 (Tibetan)', 81: '1 (Tibetan)', 82: '2 (Tibetan)', 83: '3 (Tibetan)', 84: '4 (Tibetan)', 85: '5 (Tibetan)', 86: '6 (Tibetan)', 87: '7 (Tibetan)', 88: '8 (Tibetan)', 89: '9 (Tibetan)',
    90: '0 (Urdu)', 91: '1 (Urdu)', 92: '2 (Urdu)', 93: '3 (Urdu)', 94: '4 (Urdu)', 95: '5 (Urdu)', 96: '6 (Urdu)', 97: '7 (Urdu)', 98: '8 (Urdu)', 99: '9 (Urdu)'
}
image = np.squeeze(image)
plt.imshow(image, cmap='gray')
#actual_class = np.argmax(y_test[20])
plt.title(f"Predicted: {digit_language_mapping[predicted_class]}, actual: 4 in arabic")
plt.show()
