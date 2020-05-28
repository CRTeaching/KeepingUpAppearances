import os, sys
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
import numpy as np

from keras import models
import json

from skimage.io import imsave

"""
Provided a folder, load a number of images equal to batch_size.
Save it in a given "batch".
"""
def load_images(batch_size, batch, folder='static/Train/'):
    try:
        number_of_files = 0
        for filename in os.listdir(folder):
            temporary_img = load_img(folder + filename)
            temporary_img = img_to_array(temporary_img)
            batch.append(temporary_img) #Add the image to the batch
            number_of_files += 1 # Count the number of files
            if number_of_files % 10 == 0:
                print(number_of_files) #Print every 10th number so it doesn't look stuck.
            # if the batch_size is met or exceeded, end the function loop.
            if number_of_files >= batch_size:
                break
        # Convert standard array to numpy array - can take a second or two
        batch = np.array(batch)#, dtype=float) #float gives error, use 1.0/255 instead.
    except:
        print("It is possible that the Train folder is not in its supposed location!")
        print("Ensure that the folder param is filled in correctly, or that the Train/ set matches the default location('static/Train/')")
        sys.exit(0)
    return batch

# Generate training data
def image_a_b_gen(datagen, Xtrain):
    #for every picture: get its L and ab channels
    #and save them for later
    try:
        for batch in datagen.flow(Xtrain):#, batch_size=batch_size):
            lab_batch = rgb2lab(batch)
            X_batch = lab_batch[:,:,:,0] # get the lightness channel
            Y_batch = lab_batch[:,:,:,1:] / 128
            # yield is like return but it just keeps the variables
            # in the memory, that way it doesn't stop the for loop on 1st
            # loop.
            yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)
            # it might be the cause of the out of memory error.
    except:
        print("Ensure that the input given is of correct format!")
        sys.exit(0) # if this breaks, it will be inside the training loop, in which case, close the program.
"""
This function trains the model.
Based on my experience, some computers support multiprocessing
but dont support callbacks, while others do vice versa.
Callback isn't that useful to the end user so the option to turn it on
(debugging) is by default False and can be turned on if somebody wishes so.
"""
def train(datagen, Xtrain, model, steps, epochs_given, debugging = True):
    try:
        # In my experience, callback only works on Manjaro.
        if(debugging):
            model.fit_generator(image_a_b_gen(datagen, Xtrain),
                                steps_per_epoch=steps,
                                epochs=epochs_given)#,use_multiprocessing=True)
        else:
            from keras.callbacks import TensorBoard
            tensorboard = TensorBoard(log_dir="output/current_run")
            tensorboard = [tensorboard]
            model.fit_generator(image_a_b_gen(datagen, Xtrain),
                                steps_per_epoch=3,
                                epochs=1,
                                callbacks=tensorboard)
    except:
        print("Something happened during the training loop.")
        print("Try turning debbuing to True if you have disabled it.")
        print("Ensure that the model is loaded in, Xtrain split has happened, and steps and epochs value are correct")
        sys.exit(0) # quit the program if the loop breaks.
"""
Save the model and its weights.
Allows users to specify the name of the model if they
don't want to override their previous work.
"""
def save_model(model, name="model"):
    try:
        model_json = model.to_json()
        with open(name + ".json", "w") as json_file:
            #json_file.write(model_json) #json.dump
            json.dump(model_json, json_file)
        model.save_weights(name + ".hdf5")
    except:
        print("Something went wrong while saving the model.\nTry using .write() instead of .dump() if the problem persists")
        # An error while saving the model should not close the program down!

def loadModel(name):
    try:
        with open(name+'.json','r') as f:
            model_json = json.load(f)
        model = models.model_from_json(model_json)
        model.load_weights(name+'.hdf5')
        #model = models.load_model(name)
        return model
    except:
        print("Something went wrong while loading the model, ensure that the name given is correct.")
        print("As well as the model is in the main directory of the repository.")
        print("Both, the .hdf5 and .json files are needed to load the model correctly.")
        # An error while loading the model should not close the program down!

"""
Although we measure the model's accuracy with test_accurracy,
It is best to see how accurate the neural network is by actually seeing
what it can produce.
This function is the main method for evaluating the model's performance.
A default folder with the testing images is given but can be changed accordingly.
"""
def prepare_accuracy_visualisation_images(color_me, folder='static/Test/'):
    try:
        for filename in os.listdir(folder):
            color_me.append(img_to_array(load_img(folder+filename)))
        color_me = np.array(color_me, dtype=float)
        color_me = rgb2lab(1.0/255 * color_me)[:,:,:,0]
        color_me = color_me.reshape(color_me.shape+(1,))
        return color_me
    except:
        print("Something went wrong while pre-processing the data.")
        print("Ensure the pictures are in the correct folder, appropriate image sizes are given and try again.")
        sys.exit(0)

"""
Save the images.
"""
def save_the_images(output, color_me, folder="static/Result/"):
    try:
        # Add a module that does float64 to uint8 conversion for us
        from skimage import img_as_ubyte
        
        # Output colorization
        for i in range(len(output)):
            # Create an empty matrix (3 channels, 256 by 256 pixels)
            create_image = np.zeros((256, 256, 3))
            # Get the lightness channel from the output
            create_image[:,:,0] = color_me[i][:,:,0]
            # Get the ab channels from output.
            create_image[:,:,1:] = output[i]

            create_image = lab2rgb(create_image)
            # To avoid lossy conversion, convert float64 to uint8.
            create_image = img_as_ubyte(create_image)

            imsave(folder+"img_"+str(i)+".png", create_image)
            print("Saved picture number: ", i)
    except:
        print("While saving the images, an error occured.")
        print("Ensure the folder where you want to save them exists (default: 'static/Result/').")
        sys.exit(0)