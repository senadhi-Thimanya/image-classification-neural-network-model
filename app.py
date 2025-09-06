import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Rescaling
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import os
from huggingface_hub import hf_hub_download

st.header("Image Classification Model")

# Load model from Hugging Face
model_path = hf_hub_download(
    repo_id="Raemih/fruit-veg-img-classifier",
    filename="Image_classify.keras",
    token=os.environ.get("HF_TOKEN")
)
model = tf.keras.models.load_model(model_path)

data_cat = [
    'apple','banana','beetroot','bell pepper','cabbage','capsicum','carrot',
    'cauliflower','chilli pepper','corn','cucumber','eggplant','garlic','ginger',
    'grapes','jalepeno','kiwi','lemon','lettuce','mango','onion','orange','paprika',
    'pear','peas','pineapple','pomegranate','potato','raddish','soy beans','spinach',
    'sweetcorn','sweetpotato','tomato','turnip','watermelon'
]

img_height = 180
img_width = 180

# Hybrid input: File uploader OR URL
uploaded_file = st.file_uploader("Upload or paste an image", type=["jpg", "jpeg", "png"])
image_url = st.text_input("Or paste an image URL")

img = None

if uploaded_file is not None:
    # File uploader case
    img = Image.open(uploaded_file).convert("RGB")

elif image_url:
    # URL case
    try:
        r = requests.get(image_url, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
    except Exception as e:
        st.error(f"Could not load image from URL. {e}")

# If we have an image, process it
if img:
    st.image(img, caption="Input Image", width=220)

    # --- Preprocess (NO manual /255.0 if model already rescales) ---
    arr = tf.keras.utils.img_to_array(img.resize((img_width, img_height))).astype("float32")
    arr = np.expand_dims(arr, axis=0)  # shape: (1, H, W, 3)

    # Optional: only normalize if model does NOT have a Rescaling layer
    has_rescale = any(isinstance(l, Rescaling) for l in model.layers)
    if not has_rescale:
        arr = arr / 255.0

    preds = model.predict(arr)
    probs = tf.nn.softmax(preds[0]).numpy()

    top = int(np.argmax(probs))
    st.subheader(f"Predicted: {data_cat[top]}")
    st.write(f"Confidence: {probs[top]*100:.2f}%")

    # Show top-3 predictions
    top3 = np.argsort(-probs)[:3]
    st.caption("Top-3 Predictions:")
    for k in top3:
        st.write(f"- {data_cat[k]}: {probs[k]*100:.2f}%")
