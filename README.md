# Hand Motion Recognition

A Python-based real-time **hand gesture recognition** project using **OpenCV**, **MediaPipe**, and **TensorFlow/Keras**. This project detects hand landmarks from a webcam feed, extracts features, and classifies gestures such as `fist`, `thumb`, `victory`, and more.

---

## Features

- Real-time hand detection using **MediaPipe Hands**  
- Feature extraction from **hand landmarks**  
- Trainable **neural network model** for static gestures (MLP)  
- Save and load gesture data in **JSON format**  
- Real-time gesture prediction overlay on webcam feed  
- GPU/CPU compatible (TensorFlow automatically uses available GPU)  

---

## Project Structure

HANDTRACKINGPROJECT/
│
├─ dataset_json/ # Folder for saving hand landmarks per gesture
├─ utils.py # Feature extraction utilities + script for collecting landmark data
├─ model.py # Neural network class for training and prediction
├─ main.py # Script for live gesture recognition
├─ .gitignore
└─ README.md

---

## Collect Gesture Data

Run main.py to save hand landmarks:
Press keys (e.g., 1, 2, 3) to select the gesture label.
Press q to quit.

---

## Real-Time Gesture Recognition

Run main.py and press 'p' to predict the real-time gesture.

Built by Bezhan Sayfiev Copyright © by Bezhan Sayfiev for his personal purposes. You are 100% allowed to use this project for both personal and commercial use, but NOT to claim it as your own design and work. A credit to the original author, Bezhan Sayfiev, is of course highly appreciated!