# Girlfriend Emotion Detection
Detects my girlfriend's emotions from facial patterns using a image CNN and outputs the emotive result live.  

## Overview

This project focuses on implementing an Emotion Detection system using deep learning techniques. The model is trained to recognize facial expressions in images, categorizing emotions such as anger, disgust, fear, happiness, sadness, surprise, and neutral.

## Project Structure

The project is organized into several components:

1. **Model Training:**
    - The model is built using the MobileNet architecture, with a custom output layer for emotion classification.
    - The training script `emotion_detection.ipynb` preprocesses and augments the training data, compiles the model, and trains it on a dataset of facial images with corresponding emotion labels.

2. **Data Preprocessing:**
    - The training and validation datasets are prepared using the Keras ImageDataGenerator. Training data is augmented with techniques like zooming, shearing, and horizontal flipping. Validation data is rescaled.

3. **Visualization:**
    - The script `emotion_detection.ipynb` includes functions to visualize a sample of training images and their corresponding labels.

4. **Callbacks:**
    - The model utilizes callbacks, such as early stopping and model checkpointing, to enhance training efficiency and save the best model based on validation accuracy.

5. **Model Evaluation:**
    - The model's performance is evaluated on the validation dataset, and the training history is saved for further analysis.

6. **Emotion Prediction on Images:**
    - The script `detection_test.ipynb` loads the trained model and predicts the emotion of a given input image.

7. **Real-time Emotion Detection:**
    - The real-time emotion detection script `live_analysis.py` uses OpenCV to capture video frames from a webcam. It detects faces in the frames using Haar cascades and predicts emotions on the detected faces using the trained model.

## How to Use

1. **Install Dependencies:**
    - Install the required libraries by running `pip install -r requirements.txt`.

2. **Train the Model:**
    - Execute the training function in `emotion_detection.ipynb`.

3. **Visualize Training Images:**
    - Run the vizualization script in `emotion_detection.ipynb` to visualize a sample of training images.

4. **Emotion Prediction on Singular Images:**
    - Refactor the img path in `detection_test.ipynb` to predict the emotion for a singular image.  

5. **Real-time Emotion Detection:**
    - Run `python live_analysis.py` for real-time emotion detection using your webcam.

## Files and Directories

- **`train_emotion_model.py`**: Script for training the emotion detection model.
- **`visualization.py`**: Script for visualizing training images.
- **`predict_emotion.py`**: Script for predicting emotions on single images.
- **`real_time_emotion_detection.py`**: Script for real-time emotion detection using a webcam.
- **`FacialAnalysis/TrainData/`**: Directory containing the training dataset.
- **`FacialAnalysis/TestData/`**: Directory containing the validation dataset.
- **`best_model.h5`**: Saved model checkpoint of the best-performing model during training.
- **`README.md`**: Project documentation file.

## Acknowledgments

- The emotion detection model is built on the MobileNet architecture using the Keras library.
- Haar cascades from OpenCV are used for face detection.
- The project may require adjustments based on specific use cases and dataset characteristics.

Feel free to explore, modify, and extend this project for your own applications!
