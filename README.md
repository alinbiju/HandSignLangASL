# Hand Sign Language Recognition

This repository contains code for a Hand Sign Language recognition system using Python, OpenCV, and a pre-trained deep learning model. The system can recognize hand signs corresponding to English alphabet letters (A-Z).

## Overview

The Hand Sign Language Recognition system captures video from a webcam and detects a user's hand to recognize the sign language gesture they are making. It uses computer vision techniques and a deep learning model to classify and display the corresponding alphabet letter in real-time.

## Requirements

Make sure you have the following libraries and dependencies installed:

- Python 3.x
- OpenCV (cv2)
- cvzone
- TensorFlow (for the pre-trained deep learning model)

You can install the required packages using the following command:

```bash
pip install opencv-python-headless cvzone tensorflow
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/HandSignLangASL.git
   cd HandSignLangASL
   ```

2. Run the Python script:

   ```bash
   python test.py
   ```

3. A window will open showing the video feed from your webcam. Place your hand within the frame and make one of the hand signs corresponding to English alphabet letters (A-Z).

4. The system will detect your hand, recognize the sign, and display the corresponding letter on the screen.

## Directory Structure

- `test.py`: The main Python script for hand sign recognition.
- `Model/keras_model.h5`: Pre-trained deep learning model for classification.
- `Model/labels.txt`: Text file containing the labels (A-Z) for the model.
- `Data/C`: Directory containing sample hand sign images for model training (you can replace this with your own dataset).

## Additional Information

- The `test.py` script captures the video feed from your webcam, detects your hand, and processes the image to classify the hand sign using the pre-trained deep learning model.

- The `cvzone` library is used for hand detection, and TensorFlow is used for running the pre-trained model.

- The recognized letter is displayed on the screen in real-time, and you can customize the appearance or behavior in the script as needed.

- This system is designed for educational and demonstrative purposes and can be extended for more complex sign language recognition tasks.
