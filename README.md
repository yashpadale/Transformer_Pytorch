# Transformer_Pytorch
 This is a pytorch implementation for making a language model with very few lines of code. It manages all text preprocessing and user just needs to put their data in the train.txt put in their hyper parameters in the init.py and can train ,save,predict their model .
## Overview
Transformer_PyTorch is a minimalistic PyTorch implementation of a Transformer-based language model. It simplifies the process of creating, training, and deploying language models by handling text preprocessing and other boilerplate tasks. Users only need to provide their training data and set hyperparameters to get started.
## Features
Easy Setup: Place your training data in train.txt and configure hyperparameters in init.py.
Text Preprocessing: Automatically manages tokenization and padding.
Training: Simple API for training your Transformer model.
Saving and Loading: Save and load your trained model with ease.
Prediction: Generate predictions or perform text generation tasks.

## Installation
             git clone https://github.com/yashpadale/Transformer_Pytorch
## Enter the directory
             cd Transformer_PyTorch
             cd LanguageModel
## Install the dependancies
             pip install requirements.txt

# Usage
### Preparing Your Data
Create a train.txt File:
Place your training data in train.txt. Each line should represent a training example.

## Configure Hyperparameters:
Edit init.py to set your desired hyperparameters, such as number of blocks , maxlen , batch size, number of epochs, etc.
