# Fruit_Vegetable_Recognition-ML
This project uses machine learning models to recognize and classify images of fruits and vegetables. The dataset consists of images categorized into different fruit and vegetable classes, which are divided into train, validation, and test sets. The main goals are to:

Load and visualize images from the dataset.
Train a Convolutional Neural Network (CNN) model to classify these images.
Evaluate the performance of the CNN model and compare the results.
## Table of Contents
- Dataset
- Model Training
- Convolutional Neural Network
- Result

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
- Decision Tree
- Random Forest
These models are implemented using the scikit-learn library. However, since image data requires special preprocessing and feature extraction, we apply some basic transformations such as resizing and normalization using ImageDataGenerator.

## Convolutional Neural Network (CNN)
In addition to traditional machine learning models, we also train a CNN for image classification. CNNs have been shown to perform well on image data due to their ability to automatically extract features from images.

The CNN architecture is as follows:

Conv2D layers with ReLU activation and max-pooling layers.
Fully connected layers for classification.
Softmax activation for multi-class classification.

## Result
We trained both traditional machine learning (ML) models and a Convolutional Neural Network (CNN) on the Fruit and Vegetable Image Recognition dataset and compared their performance in terms of accuracy.

- Machine Learning Models:
Logistic Regression: 96.6% accuracy
SVM: 83% accuracy
Decision Tree:96.7% accuracy
Random Forest: 96.1% accuracy
- CNN Model:
CNN: 89.2% accuracy (best-performing model)
Analysis:
The CNN outperformed all traditional ML models by a significant margin, demonstrating the effectiveness of deep learning for image classification tasks.
Random Forest performed the best among traditional models but still lagged behind the CNN.
Traditional models struggled with the complex nature of image data, while the CNN was able to automatically extract relevant features, leading to better performance.
