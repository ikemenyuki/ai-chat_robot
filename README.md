# ai-chat_robot

## Overview
This is a tiny and interesting deep learning project of an Artificial Intelligence chatting robot implemented with neural networks. Is is based on some simple data so the robot is only capable of certain response.
But it can be improved by simply enlarging the dataset.

## Installation Dependencies:
- Python 3.5 or higher
- PyTorch
- NumPy
- NLTK (Natural Language Tool Kit)
- tkinter

## Preparation:
`pip install torch, pickle, nltk, numpy, tkinter`

## Running Procedures:
```
git clone https://github.com/ikemenyuki/ai-chat_robot.git
python train_chatbot.py
python chatapp.py
```

## Mechanisms:
This project implements neural networks with 3 fully-connected layers and 2 dropout layer.

The layer is specified below:
1. a fully connected layer with 128 neurons. Used ReLU as the activation function.
2. a dropout layer with p = 0.5.
3. a fully connected layer with 64 neurons. Used ReLU as the activation function.
4. a dropout layer with p = 0.5.
5. a fully connected layer matches to the number of possible final outcomes.
6. log_softmax activation function.

It is optimized with Adam (A method for stochastic optimization). The loss is measured with NLLLoss.

The model is trained 300 epochs.

## Disclaimer
This project is based on the following website.

[Python Chatbot Project – Learn to build your first chatbot using NLTK & Keras] (https://data-flair.training/blogs/python-chatbot-project/)



