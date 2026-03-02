import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os

st.title("Material Recognition & Similarity System")
st.write("If you see this, app is running correctly.")

# =============================
# Load Model Safely
# =============================
try:
    model = tf.keras.models.load_model("fmd_model.h5")
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# =============================
# Create Feature Extractor
# =============================
try:
    feature_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer("feature_layer").output
    )
    st.write("Feature extractor ready.")
except Exception as e:
    st.error(f"Feature extractor failed: {e}")
    st.stop()

# =============================
# Load Embeddings
# =============================
try:
    embeddings = np.load("embeddings.npy")
    image_paths = np.load("image_paths.npy", allow_pickle=True)
    st.write("Embeddings loaded.")
except Exception as e:
    st.error(f"Embedding loading failed: {e}")
    st.stop()

class_names = sorted(os.listdir("image"))

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is None:
    st.info("Upload an image to start.")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    st.subheader("Prediction")
    st.write(f"Material: {class_names[predicted_index]}")
    st.write(f"Confidence: {confidence:.2f}")

    user_embedding = feature_model.predict(img_array)
    similarity = cosine_similarity(user_embedding, embeddings)

    top_k = similarity[0].argsort()[-6:][::-1]
    top_k = top_k[1:6]

    st.subheader("Top 5 Similar Images")
    cols = st.columns(5)

    for i, idx in enumerate(top_k):
        cols[i].image(image_paths[idx])