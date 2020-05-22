import os
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
import numpy as np

from skimage.io import imsave

"""
Provided a folder, load a number of images equal to batch_size.
Save it in a given "batch".
"""
def load_images(batch_size, batch, folder='Train/'):
    number_of_files = 0
    for filename in os.listdir(folder):
        temporary_img = load_img(folder + filename)
        temporary_img = img_to_array(temporary_img)
        batch.append(temporary_img) #Add the image to the batch
        number_of_files += 1 # Count the number of files
        if number_of_files % 10 == 0:
            print(number_of_files) #Print every 10th number
        # if the batch_size is met, end the function loop.
        if number_of_files == batch_size:
            break

def train_test_split(batch, training_percentage):
    split = int(training_percentage * len(batch))

    # use the given percentage for training
    Xtrain = batch[:split]

    #normalise the data (divide it by 255), while keeping it as float
    Xtrain = 1.0/255 * Xtrain
    return Xtrain

# Generate training data
def image_a_b_gen(datagen, Xtrain):
    #for every picture: get its L and ab channels
    #and save them for later
    for batch in datagen.flow(Xtrain):#, batch_size=batch_size):
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0] # get the lightness channel
        Y_batch = lab_batch[:,:,:,1:] / 128
        # yield is like return but it just keeps the variables
        # in the memory, that way it doesn't stop the for loop on 1st
        # loop.
        yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)
        # it might be the cause of the out of memory error.
"""
This function trains the model.
Based on my experience, some computers support multiprocessing
but dont support callbacks, while others do vice versa.
Callback isn't that useful to the end user so the option to turn it on
(debugging) is by default False and can be turned on if somebody wishes so.
"""
def train(datagen, Xtrain, model, debugging = False):
    # In my experience, callback only works on Manjaro.
    if(debugging):
        model.fit_generator(image_a_b_gen(datagen, Xtrain),
                            steps_per_epoch=3,
                            epochs=1,
                            use_multiprocessing=True)
    else:
        from keras.callbacks import TensorBoard
        tensorboard = TensorBoard(log_dir="output/current_run")
        tensorboard = [tensorboard]
        model.fit_generator(image_a_b_gen(datagen, Xtrain),
                            steps_per_epoch=3,
                            epochs=1,
                            callbacks=tensorboard)
"""
Save the model and its weights.
Allows users to specify the name of the model if they
don't want to override their previous work.
"""
def save_model(model, name="model"):
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(name + ".h5")

"""
Test the accuracy of the model.
A remaining percentage of the batch is used to do so to avoid getting
inflated accuracy when the same pictures that were used for training
are used for testing.
"""
def test_accurracy(batch, split_percentage_for_testing, model):
    # Get the lightness channel of an image and normalise it.
    batch_test = rgb2lab(1.0/255 * batch[split_percentage_for_testing:])[:,:,:,0]
    batch_test = batch_test.reshape(batch_test+(1,))
    #Get the ab channels of the image.
    batch_ab = rgb2lab(1.0/255 * batch[split_percentage_for_testing:])[:,:,:,1:]
    # Normalise the data to be between 1 and -1.
    batch_ab = batch_ab / 128
    # Finally, print the model's accuracy.
    print(model.evaluate(batch_test, batch_ab, batch_size=20))

"""
Although we measure the model's accuracy with test_accurracy,
It is best to see how accurate the neural network is by actually seeing
what it can produce.
This function is the main method for evaluating the model's performance.
A default folder with the testing images is given but can be changed accordingly.
"""
def prepare_accuracy_visualisation_images(images_to_be_colorised, folder='Test/'):
    for filename in os.listdir(folder):
        images_to_be_colorised.append(img_to_array(load_img(folder+filename)))
    images_to_be_colorised = np.array(images_to_be_colorised, dtype=float)
    images_to_be_colorised = rgb2lab(1.0/255 * images_to_be_colorised)[:,:,:,0]
    images_to_be_colorised = images_to_be_colorised.reshape(
        images_to_be_colorised.shape+(1,))

"""
Save the images.
"""
def save_the_images(output):
    # Add a module that does float64 to uint8 conversion for us
    from skimage import img_as_ubyte
    
    # Output colorization
    for i in range(len(output)):
        # Create an empty matrix (3 channels, 256 by 256 pixels)
        create_image = np.zeros((256, 256, 3))
        # Get the lightness channel from the output
        create_image[:,:,0] = output[i][:,:,0]
        # Get the ab channels from output.
        create_image[:,:,1] = output[i]

        create_image = lab2rgb(create_image)
        # To avoid lossy conversion, convert float64 to uint8.
        create_image = img_as_ubyte(create_image)

        imsave("Result/img_"+str(i)+".png", create_image)
        print("Saved picture number: ", i)

"""
Colorise the selected images and save them in results folder.
"""
def colorise_output(model, images_to_be_colorised):
    output = model.predict(images_to_be_colorised)
    # The neural network makes the values to be between -1 and 1.
    # To restore the values of ab channels, we multiply them by 128.
    output = output * 128
    # save the images.
    save_the_images(output)
    #return output # not needed since we save the images directly from here.
