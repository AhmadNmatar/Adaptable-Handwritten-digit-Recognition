# Adaptable Handwritten Digit Recognition

**Adaptable Handwritten Digit Recognition** is an AI system designed to recognize handwritten digits in **10 different languages** using a **Convolutional Neural Network (CNN)** and **computer vision** techniques.

The CNN is trained on the **MNIST-MIX dataset**, which includes digits from the following languages:
- Arabic
- Bangla
- Devanagari
- English
- Farsi
- Kannada
- Swedish
- Telugu
- Tibetan
- Urdu

The system processes images either uploaded manually or captured via a webcam, feeds them into the trained CNN model, and predicts the digit.

---

## Features
- **Real-time digit recognition** using webcam input.
- **Image upload support** for digit recognition.
- High accuracy across multiple languages due to training on the MNIST-MIX dataset.
- Built with **TensorFlow**, **OpenCV**, and **Python**.

---

## How to Test the System

The system includes two key scripts:
1. **`realTime.py`**  
   This script performs real-time digit recognition using a webcam. It:
   - Captures input via webcam.
   - Processes the input by detecting contours.
   - Predicts the digit using the CNN model.
   - Displays the result on the screen in real-time.

2. **`image_upload.py`**  
   This script allows digit recognition from an uploaded image. It:
   - Processes the uploaded image.
   - Feeds the processed image into the CNN model for prediction.
   - Displays the predicted result.

---

### Steps to Set Up and Run the System

  1. Clone the repository to your local machine:
       ```bash
       git clone https://github.com/AhmadNmatar/Adaptable-Handwritten-digit-Recognition.git ```
    
  2. Navigate to the project directory
       ```bash
           cd Adaptable-Handwritten-digit-Recognition
    
  3. Create a Python virtual environment and activate it
      ```bash
          python3 -m venv your_env_name
          source your_env_name/bin/activate 
    
    
    
  4. Install the required libraries:
     ```bash
     pip install tensorflow
     pip install opencv-python
     pip install numpy
     pip install matplotlib
    
    
  5. Verify the installation of libraries:
      ```bash
        pip list
    
  6.Run the desired script:
  
  * For real-time recognition:
    ```bash
      python realTime.py

  * For image upload recognition:
    ```bash
    python image_upload.py

  
