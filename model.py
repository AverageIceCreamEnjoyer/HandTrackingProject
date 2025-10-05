import os
import numpy as np
import json
from typing import Tuple, List

from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

MODEL_NAME = "hand_sign_model_1.h5"

class HandSignModel:
    def __init__(self, input_dim=None, num_classes=None, model_path=None):
        """
        Initialize a new model or load an existing one.
        - input_dim: number of features per sample (e.g., 78)
        - num_classes: number of gesture classes
        - model_path: optional, path to load an existing model (.h5)
        """
        self.label_encoder = None
        self.model = None

        # Choose device
        if tf.config.list_physical_devices('GPU'):
            self.device = "/GPU:0"
        else:
            self.device = "/CPU:0"

        with tf.device(self.device):
            if model_path and os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                self.model = keras.models.load_model(model_path)
            elif input_dim and num_classes:
                print("Creating a new model...")
                self.model = keras.Sequential([
                    layers.Dense(128, activation="relu", input_shape=(input_dim,)),
                    layers.Dropout(0.3),
                    layers.Dense(64, activation="relu"),
                    layers.Dropout(0.3),
                    layers.Dense(num_classes, activation="softmax")
                ])
                self.model.compile(optimizer="adam",
                                loss="categorical_crossentropy",
                                metrics=["accuracy"])
            else:
                raise ValueError("Provide either (input_dim & num_classes) or model_path.")

    def fit(self, X, y, epochs=30, batch_size=32, val_split=0.2):
        """
        Train the model on given data.
        - X: numpy array of shape (samples, features)
        - y: list/array of labels (strings)
        """
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = keras.utils.to_categorical(y_encoded)
        with tf.device(self.device):
            history = self.model.fit(X, y_categorical,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    validation_split=val_split)
        return history

    def predict(self, X):
        """
        Predict labels for given feature vectors.
        - X: numpy array of shape (samples, features)
        Returns list of predicted labels (strings).
        """
        with tf.device(self.device):
            probs = self.model.predict(X)
        preds = np.argmax(probs, axis=1)
        if self.label_encoder:
            return self.label_encoder.inverse_transform(preds)
        else:
            return preds  # raw numeric if encoder missing

    def save(self, path=MODEL_NAME):
        """Save trained model and label encoder."""
        self.model.save(path)
        if self.label_encoder:
            classes_path = path.replace(".h5", "_classes.npy")
            np.save(classes_path, self.label_encoder.classes_)
            print(f"Saved model to {path} and classes to {classes_path}")

    def load_classes(self, classes_path):
        """Load label encoder classes after reloading model."""
        classes = np.load(classes_path, allow_pickle=True)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = classes



def load_dataset(data_dir: str) -> Tuple[np.array, list]:
    # Load dataset (assuming JSON files per label in dataset_json/)
    X, y = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            label = filename.replace(".json", "")
            with open(os.path.join(data_dir, filename)) as f:
                samples = json.load(f)
            for feat in samples:
                X.append(feat)
                y.append(label)

    X = np.array(X, dtype=np.float32)

    return X, y