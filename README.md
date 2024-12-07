# Fruit_Vegetable_Recognition-ML
This project uses machine learning models to recognize and classify images of fruits and vegetables. The dataset consists of images categorized into different fruit and vegetable classes, which are divided into train, validation, and test sets. The main goals are to:

Load and visualize images from the dataset.
Train a Convolutional Neural Network (CNN) model to classify these images.
Evaluate the performance of the CNN model and compare the results.
## Table of Contents
- Installation
- Dataset
- Data Visualization
- Model Training
- Model Evaluation
- Results
- License

## Dataset
The dataset for this project, "Fruit and Vegetable Image Recognition," contains images of various fruits and vegetables categorized into different classes. The dataset is structured as follows:

- train/: Contains images for training the model.
- test/: Contains images for testing the model's performance.
- validation/: Used for validating the model during training.

## Model Training
Machine Learning Models
For the classification task, we will start by training several machine learning models, including:

- Logistic Regression
- Support Vector Machine (SVM)
- k-Nearest Neighbors (k-NN)
- Random Forest
These models are implemented using the scikit-learn library. However, since image data requires special preprocessing and feature extraction, we apply some basic transformations such as resizing and normalization using ImageDataGenerator.

## Convolutional Neural Network (CNN)
In addition to traditional machine learning models, we also train a CNN for image classification. CNNs have been shown to perform well on image data due to their ability to automatically extract features from images.

The CNN architecture is as follows:

Conv2D layers with ReLU activation and max-pooling layers.
Fully connected layers for classification.
Softmax activation for multi-class classification.
