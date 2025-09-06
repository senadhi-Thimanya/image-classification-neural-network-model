# Fruits and Vegetables Image Classification Model

This project demonstrates building a deep learning-based image classification model using TensorFlow and Keras to classify images of fruits and vegetables. The model is trained on a dataset of 36 classes of fruits and vegetables, achieves high accuracy, and is deployed as a simple web application using Streamlit.

<img width="1908" height="1522" alt="img" src="https://github.com/user-attachments/assets/c0ac4f3e-11fd-4769-bfbf-d58590128429" />

Demo: https://image-classification-neural-network-model.streamlit.app/
Model: https://huggingface.co/Raemih/fruit-veg-img-classifier

## Credit to Original Tutorial

This project is initially based on the YouTube tutorial by [KothaEd](https://www.youtube.com/channel/UCZ0RhgwvVWKwK79rZCLYTpg): [Image Classification Deep Learning Neural Network Model in Python with TensorFlow](https://www.youtube.com/watch?v=V61xy1ZnVTM). The tutorial provided the foundational steps for data preprocessing, model building, training, and initial Streamlit deployment. Subsequent enhancements were made to improve accessibility, model performance, and user experience.

## Project Overview

- **Objective**: Classify images into 36 categories of fruits and vegetables (e.g., apple, banana, cabbage, chili, etc.).
- **Approach**:
  - Preprocess image data from folders into arrays using TensorFlow.
  - Build a sequential Convolutional Neural Network (CNN) model with multiple layers.
  - Train the model using training and validation datasets with data augmentation.
  - Evaluate the model on test data and new images.
  - Deploy the model as a web app for interactive predictions, hosted online for easy access.
- **Key Features**:
  - Handles image resizing, shuffling, and batching.
  - Uses validation data and data augmentation to prevent overfitting and improve generalization.
  - Visualizes training accuracy and loss.
  - Web app supports multiple input methods: pasting image URLs, drag-and-drop uploads, or browsing local files.
  - Model hosted on Hugging Face for seamless integration.
  - Achieves Top-3 Accuracy of 0.96657383 after enhanced training.

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
  - Hugging Face Hub (for model hosting)

Install dependencies:
```
pip install tensorflow streamlit numpy pandas matplotlib huggingface_hub
```

## Project Structure

- `image-classification-model.ipynb`: Original Jupyter notebook for data preprocessing, model building, training, and evaluation.
- `image-classification-model-refine.ipynb`: Modified Jupyter notebook for data preprocessing, model building and training.
- `model-evaluation.ipynb`: Jupyter notebook for model evaluations.
- `app.py`: Streamlit web app script for deploying the model.
- `requirements.txt`: Requirements to install when running the Streamlit web app.
- `.env`: Add the huggingface access token.
- `image_classifier.keras`: Saved trained model file.
- `fruits_vegetables/`: Dataset folder (not included in repo; download separately).
- `.venv/`: Python kernel environment for the jupyter notebooks to run (not included in the repo; create seperately.)
- `model-to-hf.py`: Python file used to pass the model to huggingface.

## Workflow

### 1. Data Preprocessing

- Load images from the `fruits_vegetables` dataset folders.
- Resize images to a consistent size (180x180).
- Apply data augmentation techniques (e.g., random flips, rotations, zooms) to increase dataset diversity and improve model robustness.
- Shuffle and batch the data for efficient training.
- Split into training, validation, and test sets.

### 2. Model Building and Training

- Build a sequential CNN model:
  - Rescaling layer (divide by 255).
  - 3 Conv2D + MaxPooling2D layers (16, 32, 64 filters).
  - Flatten layer.
  - Dropout (0.2).
  - Dense layers (128 units, then 36 output units with softmax).
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy (including Top-3 Accuracy for evaluation)
- Train for 39 epochs with data augmentation.
- Visualize accuracy and loss curves to monitor performance.
- Save the model as `image_classifier.keras` and upload to Hugging Face for version control and easy access.

### 3. Model Evaluation and Testing

- load the test data
- run the `model-evaluation.ipynb` file to get the evaluation results of matrices:
  - Confusion matrix
  - precision/recall/f1 score
  - Top-3 accuracy


After training, the model achieves ~95-98% training accuracy, ~85-95% validation accuracy, and a Top-3 Accuracy of 0.96657383 on test data.

### 4. Deployment as Web App

- The app is deployed online via Streamlit Sharing, accessible without cloning the repository.
- Model is loaded from Hugging Face using an access token for security and reliability.
- User Input Options:
  - Paste an image URL (e.g., from a Google search).
  - Drag and drop an image file.
  - Browse and upload a local image file.
- The app processes the input, displays the image, and outputs the predicted class with accuracy (e.g., "Fruit in image is apple with an accuracy of 99.95%").

## To run locally (optional):

- Download the dataset [Google Drive Dataset](https://drive.google.com/file/d/1CGiAWso43GCsNo_faRq4jdDIlmwy7YI4/view?usp=sharing)
- Clone the repository to VS Code
```
git clone https://github.com/senadhi-Thimanya/image-classification-neural-network-model.git
```
- Create a python virtual environment (kernal to run the ipynb files)
- Run `image-classification-model-refine.ipynb` and save the model
- Run `model-to-hf.py` to pass the model to huggingface
- run the streamlit web app command on the terminal, inside the .venv 
```
streamlit run app.py
```
Access at http://localhost:8501 or connect streamlit to your github repository and upload the app to use online.

### For new images:

No need to download or push to GitHub; use URL paste, drag-and-drop, or browse directly in the app.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
