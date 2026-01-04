import streamlit as st
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

def load_css(file_name):
    if os.path.exists(file_name):  # Check if file exists
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.error(f"CSS file '{file_name}' not found. Make sure it exists!")
load_css("style.css")

# Function to extract color histogram
def extract_color_histogram(image_path, bins=(16, 16, 16)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Function to load dataset
def load_dataset(folder_path):
    X, y = [], []
    for label in os.listdir(folder_path):
        label_folder = os.path.join(folder_path, label)
        if os.path.isdir(label_folder):
            for img_file in os.listdir(label_folder):
                img_path = os.path.join(label_folder, img_file)
                features = extract_color_histogram(img_path)
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)

# Streamlit UI
st.title("Flower Classification with KNN")

dataset_path = st.text_input("Enter dataset path:", "./HoaVietNam")

if st.button("Load Dataset"):
    X_train, y_train = load_dataset(os.path.join(dataset_path, "train"))
    X_test, y_test = load_dataset(os.path.join(dataset_path, "test"))
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    
    st.success("Dataset loaded successfully!")
    
    # Train KNN Model
    model = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    st.write(f"Model Accuracy: {acc:.4f}")

# Upload Image for Prediction
uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_path = f"temp.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    features = extract_color_histogram(file_path).reshape(1, -1)
    label = model.predict(features)[0]
    label_name = label_encoder.inverse_transform([label])[0]
    
    st.image(uploaded_file, caption=f"Predicted: {label_name}", use_column_width=True)
