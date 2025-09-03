# Fruits and Vegetables Image Classification Model

This project demonstrates building a deep learning-based image classification model using TensorFlow and Keras to classify images of fruits and vegetables. The model is trained on a dataset of 36 classes of fruits and vegetables, achieves high accuracy, and is deployed as a simple web application using Streamlit.

Demo: https://image-classification-neural-network-model.streamlit.app/

<img width="1908" height="1416" alt="webapp" src="https://github.com/user-attachments/assets/87ffaab0-8def-440d-8690-3a980835ac54" />

The project is based on the YouTube tutorial by [KothaEd](https://www.youtube.com/channel/UCZ0RhgwvVWKwK79rZCLYTpg): [Image Classification Deep Learning Neural Network Model in Python with TensorFlow](https://www.youtube.com/watch?v=V61xy1ZnVTM).

## Project Overview

- **Objective**: Classify images into 36 categories of fruits and vegetables (e.g., apple, banana, cabbage, chili, etc.).
- **Approach**:
  - Preprocess image data from folders into arrays using TensorFlow.
  - Build a sequential Convolutional Neural Network (CNN) model with multiple layers.
  - Train the model using training and validation datasets.
  - Evaluate the model on test data and new images.
  - Deploy the model as a web app for interactive predictions.
- **Key Features**:
  - Handles image resizing, shuffling, and batching.
  - Uses validation data to prevent overfitting.
  - Visualizes training accuracy and loss.
  - Web app allows users to input image names and view predictions with accuracy scores.

## Dataset

The dataset consists of images of 36 fruits and vegetables, split into:
- **Train**: ~3,115 images
- **Validation**: ~351 images
- **Test**: ~359 images

Each category (e.g., "apple", "banana") has its own subfolder.

- **Download Link**: [Google Drive Dataset](https://drive.google.com/file/d/1CGiAWso43GCsNo_faRq4jdDIlmwy7YI4/view?usp=sharing)
- **Classes**: ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

Place the dataset in a folder named `fruits_vegetables` with subfolders for `train`, `test`, and `validation`.

## Requirements

- Python 3.8+
- Libraries:
  - TensorFlow
  - Keras (included in TensorFlow)
  - NumPy
  - Pandas
  - Matplotlib
  - Streamlit

Install dependencies:
```
pip install tensorflow streamlit numpy pandas matplotlib
```

## Project Structure

- `image_classification_model.ipynb`: Jupyter notebook for data preprocessing, model building, training, and evaluation.
- `app.py`: Streamlit web app script for deploying the model.
- `image_classifier.keras`: Saved trained model file.
- `fruits_vegetables/`: Dataset folder (not included in repo; download separately).
- `images/`: Folder for test images (e.g., apple.jpg, banana.jpg).
- `README.md`: This file.

## Usage

### 1. Training the Model
- Open `image_classification_model.ipynb` in Jupyter Lab/Notebook or VS Code.
- Update paths to the dataset folders.
- Run the cells to:
  - Load and preprocess data.
  - Build the sequential CNN model.
  - Train for 25 epochs.
  - Visualize accuracy and loss.
  - Save the model as `image_classifier.keras`.

Model Architecture Summary:
- Rescaling layer (divide by 255).
- 3 Conv2D + MaxPooling2D layers (16, 32, 64 filters).
- Flatten layer.
- Dropout (0.2).
- Dense layers (128 units, then 36 output units with softmax).
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy

After 25 epochs, expect ~95-98% training accuracy and ~80-90% validation accuracy.

### 2. Testing the Model
- In the notebook, load test images and predict using the model.
- Example: Predicts "apple" with 99.95% accuracy on sample images.

### 3. Deploying the Web App
- Run the Streamlit app:
  ```
  streamlit run app.py
  ```
- Open the local URL (e.g., http://localhost:8501).
- Enter an image name (e.g., "apple.jpg" from the `images/` folder).
- The app displays the image and predicts the class with accuracy (e.g., "Fruit in image is apple with an accuracy of 99.95%").

For new images:
- Download images from Google or elsewhere.
- Place them in the `images/` folder.
- Input the filename in the app.

## Video Breakdown
The tutorial video covers:
- 0:00 - 0:07: Project Intro
- 0:08 - 01:26: Demo
- 01:26 - 03:07: Model Explanation
- 03:07 - 16:52: Data Preprocessing
- 16:52 - 33:55: Model Creation and Training
- 33:55 - 40:19: Predictions
- 40:20 - 53:59: Web App Deployment

## Credits
- Tutorial by [KothaEd](https://www.youtube.com/channel/UCZ0RhgwvVWKwK79rZCLYTpg)
- Code Link from Video: [Google Drive Code](https://drive.google.com/file/d/1rIXgFJLW-I2oUNRi1TAAA3yWmz053lkk/view?usp=sharing)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
