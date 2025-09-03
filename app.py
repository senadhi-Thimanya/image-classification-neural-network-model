import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Rescaling
from PIL import Image
import numpy as np
import requests
from io import BytesIO

st.header("Image Classification Model")

@st.cache_resource
def get_model():
    return load_model("Image_classify.keras")

model = get_model()

data_cat = ['apple','banana','beetroot','bell pepper','cabbage','capsicum','carrot',
 'cauliflower','chilli pepper','corn','cucumber','eggplant','garlic','ginger',
 'grapes','jalepeno','kiwi','lemon','lettuce','mango','onion','orange','paprika',
 'pear','peas','pineapple','pomegranate','potato','raddish','soy beans','spinach',
 'sweetcorn','sweetpotato','tomato','turnip','watermelon']

img_height = 180
img_width = 180

image_url = st.text_input("Enter Image URL")

def fetch_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

if image_url:
    try:
        img = fetch_image(image_url)
        st.image(img, caption="Input Image", width=220)

        # --- Preprocess (NO manual /255.0 if model already rescales) ---
        arr = tf.keras.utils.img_to_array(img.resize((img_width, img_height))).astype("float32")
        arr = np.expand_dims(arr, axis=0)  # shape: (1, H, W, 3)

        # Optional: only normalize here if your model does NOT have a Rescaling layer
        has_rescale = any(isinstance(l, Rescaling) for l in model.layers)
        if not has_rescale:
            arr = arr / 255.0

        preds = model.predict(arr)
        # If your last layer has no softmax (logits), apply softmax here:
        probs = tf.nn.softmax(preds[0]).numpy()

        top = int(np.argmax(probs))
        st.subheader(f"Predicted: {data_cat[top]}")
        st.write(f"Confidence: {probs[top]*100:.2f}%")

        # (nice for debugging) show top-3
        top3 = np.argsort(-probs)[:3]
        st.caption("Top-3:")
        for k in top3:
            st.write(f"- {data_cat[k]}: {probs[k]*100:.2f}%")

    except Exception as e:
        st.error(f"Could not process the image. {e}")
