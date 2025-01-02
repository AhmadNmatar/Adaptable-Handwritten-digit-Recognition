import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import signal
import sys

# Constants
MODEL_PATH = '/Users/ahmadmatar/Desktop/AIcourse/proj.test/data/full_model95.tflite'
WITDTH, HIEGH = 640, 480  # Webcam resolution
IMAGE_BORDER = 40  # Border width
PROB_THRESHOLD = 0.7  # Minimum confidence threshold
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# Load the TensorFlow Lite model
def load_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']  # Expected shape: (1, 28, 28, 1)
    print("Model loaded. Input shape:", input_shape)
    return interpreter, input_details, output_details


# Preprocess the image for the model
def preprocess_image(img, input_size):
    h, w = input_size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=-1)   # Add channel dimension (28, 28, 1)
    return np.expand_dims(img, axis=0)   # Add batch dimension (1, 28, 28, 1)

# Predict the digit and language
def predict(interpreter, input_details, output_details, img):
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predicted = interpreter.get_tensor(output_details[0]['index']).flatten()
    label = predicted.argmax(axis=0)
    prob = predicted[label]
    return label, prob

# Draw the prediction on the frame
def draw_prediction(frame, x, y, w, h, label, prob, mapping):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    text = f"{mapping[label]} ({prob:.2f})"
    cv2.putText(frame, text, (x + w // 5, y - h // 5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

# Main detection loop
def main():
    global cap
    interpreter, input_details, output_details = load_model(MODEL_PATH)
    INPUT_H, INPUT_W = input_details[0]['shape'][1:3]
    cap = cv2.VideoCapture(0)  
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WITDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HIEGH)
    
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

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame. Exiting...")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, frame_binary = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        frame_binary = cv2.morphologyEx(frame_binary, cv2.MORPH_CLOSE, MORPH_KERNEL)
        contours, _ = cv2.findContours(frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x < IMAGE_BORDER or x + w > WITDTH - IMAGE_BORDER or y < IMAGE_BORDER or y + h > HIEGH - IMAGE_BORDER:
                continue
            if w < INPUT_W // 2 or h < INPUT_H // 2 or w > WITDTH // 2 or h > HIEGH // 2:
                continue

            img = frame_binary[y:y + h, x:x + w]
            r = max(w, h)
            y_pad = ((w - h) // 2 if w > h else 0) + r // 5
            x_pad = ((h - w) // 2 if h > w else 0) + r // 5
            img = cv2.copyMakeBorder(img, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad,
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            img = preprocess_image(img, (INPUT_W, INPUT_H))
            label, prob = predict(interpreter, input_details, output_details, img)

            if prob >= PROB_THRESHOLD:
                draw_prediction(frame, x, y, w, h,label, prob, digit_language_mapping)

        cv2.imshow('MNIST Live Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break

# Run the main function
if __name__ == "__main__":
    main()