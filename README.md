# DnD Dice Classifier 

This repository contains code for a Convolutional Neural Network (CNN) model that classifies images of Dungeons and Dragons (DnD) dice. 

The model is built using TensorFlow and Keras, and trained on a dataset of images of DnD dice. 

## Project Structure

- `train_model.py`: Contains the code for training the CNN model on your dataset.
- `predict.py`: Contains the code for loading a trained model and using it to classify a new image.
- `dataset/`: Directory containing the training and testing datasets. The dataset should be organized in the following structure:

- dataset/ train/ d20/
           test/ d20/

Each subdirectory should contain images for the respective die.

## Usage

### Training the Model

1. Ensure you have the required dependencies installed:

  ```
  pip install -r requirements.txt
  ```

2. Run `train_model.py`:

  ```
  python train_model.py
  ```

The script will train the model on the images in the `dataset/train` directory and save the trained model to disk.

### Using the Model for Prediction

1. Ensure you have the required dependencies installed (if you haven't already):

  ```
  pip3 install -r requirements.txt
  ```

2. Run `predict.py` with the path to the image you want to classify:

  ```
  python3 predict.py
  ```

The script will load the trained model, process the input image, and output the predicted class of the dice.

## License

This project is licensed under the terms of the MIT license.