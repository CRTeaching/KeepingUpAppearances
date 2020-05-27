from utils import save_the_images, loadModel, prepare_accuracy_visualisation_images

# Firstly, load in the model from previous training sessions
model = loadModel('new_model')

# Secondly, load in images to colorise (they only have the lightness channel)
color_me = []
location = "Test/"
color_me = prepare_accuracy_visualisation_images(color_me, location)

# Thirdly, Colorize the loaded images
output = model.predict(color_me)
output = output * 128 # Turn the -1 to 1 values into proper Lab values.

# Finally, Save the colorized images
save_the_images(output, color_me)