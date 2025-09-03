import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO

st.header('Image Classification Model')
model = load_model('Image_classify.keras')

data_cat = ['apple','banana','beetroot','bell pepper','cabbage','capsicum','carrot',
 'cauliflower','chilli pepper','corn','cucumber','eggplant','garlic','ginger',
 'grapes','jalepeno','kiwi','lemon','lettuce','mango','onion','orange','paprika',
 'pear','peas','pineapple','pomegranate','potato','raddish','soy beans','spinach',
 'sweetcorn','sweetpotato','tomato','turnip','watermelon']

img_height = 180
img_width = 180

# Text input for URL
image_url = st.text_input('Enter Image URL')

if image_url:
    try:
        # Fetch image from the internet
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Show the image
        st.image(img, caption="Input Image", width=200)

        # Preprocess
        img_resized = img.resize((img_width, img_height))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, 0) / 255.0  # normalize

        # Predict
        predict = model.predict(img_array)
        score = tf.nn.softmax(predict[0])

        # Show result
        st.write('Veg/Fruit in image is **{}**'.format(data_cat[np.argmax(score)]))
        st.write('With accuracy of {:.2f}%'.format(100 * np.max(score)))

    except Exception as e:
        st.error(f"Could not load image. Error: {e}")
