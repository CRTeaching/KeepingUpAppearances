import numpy as np
import os 

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
batch_size = 30
X = load_images(batch_size, X)

# Split the loaded dataset into training and testing part
# as per the given percentage (90% by default)
training_percentage = 0.9
split = int(training_percentage * len(X))

# use the given percentage for training
Xtrain = X[:split]
#rnomalise the data (divide it by 255), while keeping it as float
Xtrain = 1.0/255 * Xtrain

# Load the neural network
model = ai_model()
model.summary() #TESTING THIS OUT.
from tensorflow import keras
learning_rate_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=100,
        decay_rate=0.2
)
optim = keras.optimizers.RMSprop(learning_rate=learning_rate_scheduler)

model.compile(optimizer=optim, loss='mse')

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
# Last param is optional (True by default). Enables callbacks' tensorboard if false
# In my experience it can cause errors on different systems so it is disabled by default.
steps = 5 #steps_per_epochs value
epochs_given = 5 # epochs for the training loop
#print(np.any(np.isnan(Xtrain))) # locating loss:nan issue.
train(datagen, Xtrain, model, steps, epochs_given)

# Save the selected model under the name given as second param (default: "model")
save_model(model, "model_2")

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
print(model.evaluate(Xtest, Ytest, batch_size=70))

# Load the Testing subdataset and prepare the images
color_me = []

# Second param is optional, points to the folder
# with testing images (Default: "static/Test/")
color_me = prepare_accuracy_visualisation_images(color_me)

output = model.predict(color_me)
# The neural network makes the values to be between -1 and 1.
# To restore the values of ab channels, we multiply them by 128.
output = output * 128
# save the images.
save_the_images(output, color_me)