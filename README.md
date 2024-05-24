# Fashion MNIST Classification Project

This project involves building and training a Convolutional Neural Network (CNN) and a Multilayer Perceptron (MLP) to classify images from the Fashion MNIST dataset. The project includes feature extraction using a CNN and classification using an MLP implemented from scratch.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup Instructions](#setup-instructions)
- [Running the Project](#running-the-project)
- [Results and Visualization](#results-and-visualization)
- [Acknowledgements](#acknowledgements)
- [Contributors](#contributors)
- [License](#license)

## Introduction
The goal of this project is to accurately classify images of clothing items into one of the 10 categories provided by the Fashion MNIST dataset. The project involves:
1. Building the Fully Connected Neural Network (MLP) with functions like activation, forward, backward, and cross-entropy from scratch.
2. Extracting features using a Convolutional Neural Network (CNN).
3. Classifying the extracted features using a Multilayer Perceptron (MLP).

## Dataset
The Fashion MNIST dataset consists of 70,000 grayscale images in 10 categories, with 7,000 images per category. The images are 28x28 pixels in size. The dataset is split into:
- Training set: 60,000 images (80% training set and 20% validation set)
- Test set: 10,000 images

The dataset can be downloaded and loaded using PyTorch's `torchvision.datasets` module.

## Model Architecture
### Multilayer Perceptron (MLP - functions built from scratch)
- **Input Layer**: Accepts input images of size 28x28.
- **Hidden Layer**: A single hidden layer with ReLU activation (128 neurons).
- **Output Layer**: Softmax output layer for classification.
- **Custom Implementations**: The MLP includes custom implementations for forward pass, backward pass, loss computation, and weight updates.

### Convolutional Neural Network (CNN)
- **Layers**: 5 Convolutional layers with ReLU activation and MaxPooling.
- **Regularization**: Dropout layers to prevent overfitting.
- **Feature Extraction**: The final fully connected layer outputs features for each image.

### Multilayer Perceptron (MLP) - Using Extracted Features
- **Input Layer**: Accepts features extracted by the CNN.
- **Hidden Layer**: A single hidden layer with ReLU activation.
- **Output Layer**: Softmax output layer for classification.
- **Custom Implementations**: The MLP includes custom implementations for forward pass, backward pass, loss computation, and weight updates.

## Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/your-github-username/Deep-Learning-Projects.git
cd Deep-Learning-Projects
pip install torch torchvision matplotlib scikit-learn

## Running the Project
```bash
jupyter notebook DLGroup6_FashionMnist.ipynb

[Results and Visualization](#results-and-visualization)
We getting Test accuracy upto 93%

## Acknowledgements
this project uses the Fashion MNIST dataset provided by Zalando Research and leverages PyTorch - for deep learning implementations. Special thanks to the PyTorch and scikit-learn communities for their valuable libraries and tool

## Contributors
Ruby Kumari rubykumari23@iitk.ac.in
Rupak Kumar Roy rupakroy23@iitk.ac.in 
Rahul Deval rahuldeval23@iitk.ac.in 
Sandeep Kumar sandeepkr23@iitk.ac.in
Rakesh Ranjan rrakesh23@iitk.ac.in  


## License - 

This project is not licensed - It is an assignment project- Not to be copied,with whom this link has not beeen shared.
