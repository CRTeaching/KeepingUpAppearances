import numpy as np
import os 
import random
from keras.preprocessing.image import ImageDataGenerator
from model import ai_model

# Import each method individually for clarity purposes.
from utils import load_images, save_model, save_the_images
from utils import prepare_accuracy_visualisation_images, train

# Prepare an array for images to be loaded into
X = []

# Load a number of images from a specified folder into the X array.
# 1: Batch size to load
# 2: The array to save the images in.
# 3: (Optional) Folder with the images (default: "Train/")
batch_size = 500
load_images(batch_size, X)

# Convert standard array to numpy array
X = np.array(X)#, dtype=float) #float gives error, use 1.0/255 instead.

# Split the loaded dataset into training and testing part
# as per the given percentage (80% by default)
training_percentage = 0.95
split = int(training_percentage * len(X))

# use the given percentage for training
Xtrain = X[:split]

#normalise the data (divide it by 255), while keeping it as float
Xtrain = 1.0/255 * Xtrain

# Load the neural network
model = ai_model()
from tensorflow import keras
learning_rate_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.6
)
optim = keras.optimizers.SGD(learning_rate=learning_rate_scheduler, clipnorm=1)
#optim = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optim, loss='categorical_crossentropy')

# Image transformer
# Rotate, flip, zoom in on pictures and etc so that
# the training set becomes larger
# improves the accurracy too!
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

# train the model
# Last param is optional (True by default)
# It's a boolean, disables callbacks' tensorboard if false
steps = 1000 #steps_per_epochs value
epochs_given = 100 # epochs for the training loop
train(datagen, Xtrain, model, steps, epochs_given)

# Save the selected model
# save_model(model)

## test the model
from skimage.color import rgb2lab
# Test the model using the test images
# Get the lightness channel of an image.
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
#Get the ab channels of the image.
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
# Normalise the data to be between -1 and 1.
Ytest = Ytest / 128
#print the model's accuracy.
print(model.evaluate(Xtest, Ytest, batch_size=20))

# Load the Testing subdataset and prepare the images
color_me = []

# Second param is optional, points to the folder
# with testing images (Default: "Test/")
color_me = prepare_accuracy_visualisation_images(color_me)

output = model.predict(color_me)
# The neural network makes the values to be between -1 and 1.
# To restore the values of ab channels, we multiply them by 128.
output = output * 128
# save the images.
save_the_images(output, color_me)
# conduct the testing
#colorise_output(model, color_me)