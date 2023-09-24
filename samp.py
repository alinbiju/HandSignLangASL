import cv2
import tkinter as tk
from tkinter import Label
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Create a Tkinter window
root = tk.Tk()
root.title("Sign Language Alphabet Prediction")

# Initialize the 'img' variable
img = None

# Function to update the prediction label
def update_prediction():
    global img, imgOutput
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap,:] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        prediction_label.config(text=labels[index])

    imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    imgOutput = cv2.resize(imgOutput, (800, 600))
    imgTK = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    photo = tk.PhotoImage(data=imgTK.tobytes())
    camera_label.config(image=photo)
    camera_label.image = photo
    root.after(10, update_prediction)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows (may vary depending on your OS)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
imgSize = 300

labels = ["A","B","C","D","E","F","G","H","I","K"]

# Create a label for the camera feed
camera_label = Label(root)
camera_label.pack()

# Create a label for the prediction
prediction_label = Label(root, text="", font=("Helvetica", 24))
prediction_label.pack()

# Start the update_prediction function
update_prediction()

# Run the Tkinter main loop
root.mainloop()
