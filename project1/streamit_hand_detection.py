import streamlit as st
import tensorflow.keqras
from PIL import Image
import numpy as np
import cv2

MODEL_PATH = "your_local_model_path"      # e.g., "model.h5"
LABELS_PATH = "your_local_labels_path"    # e.g., "labels.txt"

model = tensorflow.keras.models.load_model(MODEL_PATH)
labels = open(LABELS_PATH).read().splitlines()

st.title("Hand Detector (Live Camera)")

run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])

def predict(image):
    img = Image.fromarray(image).resize((224, 224))
    img_array = np.asarray(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    result = labels[np.argmax(predictions)]
    confidence = np.max(predictions)
    return result, confidence

if run:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result, confidence = predict(frame_rgb)
        cv2.putText(frame_rgb, f"{result} ({confidence:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        FRAME_WINDOW.image(frame_rgb)
    cap.release()
else:
    st.write('Camera stopped')