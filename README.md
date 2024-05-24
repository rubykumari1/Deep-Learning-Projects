# Fashion MNIST Classification Project

This project involves building and training a Convolutional Neural Network (CNN) and a Multilayer Perceptron (MLP) to classify images from the Fashion MNIST dataset. The project includes feature extraction using a CNN and classification using an MLP implemented from scratch.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset - management)
- [Model Architecture](#model-architecture)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Results and Visualization](#results-and-visualization)
- [Acknowledgements](#acknowledgements)
- [Contributors](#contributors)
- [License](#license)

## Introduction
The goal of this project is to accurately classify images of clothing items into one of the 10 categories provided by the Fashion MNIST dataset. The project involves:
1. Building the Fully connected NN - MLP with functions like activation, forward, backward, cross entropy from scratch
2. Extracting features using a Convolutional Neural Network (CNN).
3. Classifying the extracted features using a Multilayer Perceptron (MLP).

## Dataset
The Fashion MNIST dataset consists of 70,000 grayscale images in 10 categories, with 7,000 images per category. The images are 28x28 pixels in size. The dataset is split into:
- Training set: 60,000 images - 80%(training set) and 20%(validation set)
- Test set: 10,000 images

The dataset can be downloaded and loaded using PyTorch's `torchvision.datasets` module.

## Model Architecture
### Multilayer Perceptron (MLP- functions built from scratch)
- **Input Layer**: Accepts input images 28*28 
- **Hidden Layer**: A single hidden layer with ReLU activation.[128 neurons]
- **Output Layer**: Softmax output layer for classification.
- **Custom Implementations**: The MLP includes custom implementations for forward pass, backward pass, loss computation, and weight updates.

### Convolutional Neural Network (CNN)
- **Layers**: 5 Convolutional layers with ReLU activation and MaxPooling.
- **Regularization**: Dropout layers to prevent overfitting.
- **Feature Extraction**: The final fully connected layer outputs features for each image.

### Multilayer Perceptron (MLP)
- **Input Layer**: Accepts features extracted by the CNN.
- **Hidden Layer**: A single hidden layer with ReLU activation.
- **Output Layer**: Softmax output layer for classification.
- **Custom Implementations**: The MLP includes custom implementations for forward pass, backward pass, loss computation, and weight updates.

## Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/your-github-username/Deep-Learning-Projects.git
cd Deep-Learning-Projects

### 2. Install required libraries
pip install torch torchvision matplotlib scikit-learn

jupyter notebook DL_Group6_FashionMnist_DeepLearning_NeuralNet.ipynb

### Running the Project
jupyter notebook DL_Group6_FashionMnist_DeepLearning_NeuralNet.ipynb

* Download and Load the Fashion MNIST Dataset
    * The dataset is downloaded and loaded using the torchvision.datasets module.
    * The images are normalized to have values between 0 and 1.
* Split the Dataset
    * The dataset is split into training, validation, and test sets.
    * Training set: 80% of the original dataset.
    * Validation set: 20% of the original dataset.
    * Test set: 10,000 images from the test dataset.

2. **Training the CNN Model**
* Feature Extraction
    * Train the CNN model for feature extraction from the images.
    * Save the extracted features for later use with the MLP.
3. **Training the MLP Model**
* Using Extracted Features
    * Use the features extracted by the CNN to train the MLP model.
    * The MLP model is implemented from scratch, including the forward pass, backward pass, and loss computation.
* **Hyperparameter Research**
    * Experiment with different hyperparameters like learning rate, number of hidden units, and dropout rates to achieve the best performance.
    * The model uses custom weight initialization methods like He initialization for better training efficiency.
4. **Visualization**
* Training and Validation Losses
    * Plot the training and validation loss curves over epochs to observe the training process.
* Training and Validation Accuracies
    * Plot the training and validation accuracy curves over epochs to evaluate model performance.
* Confusion Matrix and Classification Report
    * Display the confusion matrix and classification report for the test dataset to understand the model's performance across different classes.
* Sample Test Image Predictions
    * Visualize the model's predictions for sample test images along with the true labels.
**Results and Visualization**


The final results include:
* Classification Accuracy: Overall accuracy on the test dataset.
* Loss Values: Training and validation losses over epochs.
* Confusion Matrix: Shows the true vs. predicted labels for the test dataset.
* Classification Report: Precision, recall, and F1-score for each class.
* Sample Predictions: Visualization of model predictions on sample test images.

**### Acknowledgements**
This project uses the Fashion MNIST dataset provided by Zalando Research and leverages PyTorch for deep learning implementations. Special thanks to the PyTorch and scikit-learn communities for their valuable libraries and tools.

**### Contributors**
* Ruby Kumari - email - rubykumari23@iitk.ac.in - github ->  @rubykumari1
* Rupak Kumar Roy  email - rupakroy23@iitk.ac.in 
* Rahul Deval email - rahuldeval23@iitk.ac.in
* Sandeep Kumar email - sandeepkr23@iitk.ac.in 
* Rakesh Ranjan email - rrakesh23@iitk.ac.in 


**### License**
This project is licensed under the MIT License - see the LICENSE file for details.

