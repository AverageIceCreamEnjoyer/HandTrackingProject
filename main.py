import cv2
import mediapipe as mp
import time
import numpy as np
from utils import landmarks_to_np, hand_size, make_feature_vector, save_sample, WRIST, MIDDLE_MCP, DATA_DIR
from model import HandSignModel, load_dataset, MODEL_NAME
import os

prev_time = 0   # Previous time
curr_time = 0   # Current time

labels = {
    '1': "fist",
    '2': "thumb",
    '3': "pinky",
    '4': "victory",
    '5': "middle"
}


# Training model
LABEL_ENCODER_NAME = MODEL_NAME.replace(".h5", "_classes.npy")
MODEL_DIR = "models"   # directory to keep models
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, LABEL_ENCODER_NAME)

if os.path.exists(MODEL_PATH):
    model = HandSignModel(model_path=MODEL_PATH)
    model.load_classes(LABEL_ENCODER_PATH)
else:
    # Traning the model
    X, y = load_dataset(DATA_DIR)
    model = HandSignModel(input_dim=X.shape[1], num_classes=len(set(y)))
    print(X.shape[1], len(set(y)))
    model.fit(X, y, epochs=20, batch_size=32)
    model.save(MODEL_PATH)



cap = cv2.VideoCapture(0)   # WebCam #0

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()

show_prediction = False
pred = ""


while True:
    success, img = cap.read()
    img_rgb =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Convert to RGB
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time

    cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    if show_prediction:
        cv2.putText(img, pred, (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)


    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    # When Esc is pressed close the window
    if key == 27:
        break
    # Store the features in a json file 
    elif chr(key) in labels:
        current_label = labels[chr(key)]
        save_sample(current_label, results.multi_hand_landmarks[0].landmark)
    # Predict the 
    elif chr(key) == "p":
        features = make_feature_vector(results.multi_hand_landmarks[0].landmark)
        features = np.array(features).reshape(1, -1)
        show_prediction = True
        pred = model.predict(features)[0]
        
cap.release()
cv2.destroyAllWindows()
