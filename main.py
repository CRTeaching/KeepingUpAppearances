import numpy as np
import os 
import random
from keras.preprocessing.image import ImageDataGenerator
from model import ai_model

from utils import load_images, train_test_split, save_model, train
from utils import test_accurracy, prepare_accuracy_visualisation_images
from utils import save_the_images, colorise_output
#import tensorflow
#from keras.layers.normalization import BatchNormalization

# Prepare an array for images to be loaded into
X = []

# Load a number of images from a specified folder into the X array.
# 1: Batch size to load
# 2: The array to save the images in.
# 3: (Optional) Folder with the images (default: "Train/")
batch_size = 50
load_images(batch_size, X)

# Convert standard array to numpy array for future
X = np.array(X)#, dtype=float) #float gives error, use 1.0/255 instead.

# Split the loaded dataset into training and testing part
# as per the given percentage (80% by default)
training_ratio = 0.8
Xtrain = train_test_split(X, training_ratio)

print("I work thus far")
# Load the neural network
model = ai_model()
model.compile(optimizer='rmsprop', loss='mse')

# Image transformer
# Rotate, flip, zoom in on pictures and etc so that
# the training set becomes larger
# improves the accurracy too!
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

print("I work thus far321")
# train the model
train(datagen, Xtrain, model)
print("I break thus far")
# Save the selected model
# save_model(model)

## test the model

# Evaluate the model's performance
testing_ratio = 1 - training_ratio
test_accurracy(X, testing_ratio, model)

# Load the Testing subdataset and prepare the images
images_to_colorise = []

# Second param is optional, points to the folder
# with testing images (Default: "Test/")
prepare_accuracy_visualisation_images(images_to_colorise)

# conduct the testing
colorise_output(model, images_to_colorise)

# save the output.

