### THIS WILL BE THE UNDERLYING API.
from flask import Flask, render_template, url_for, send_from_directory
from flask import flash, request, redirect
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/showcase/')
def show_images():
    images = []
    # Load in all the images.
    folders = os.listdir('static/Example Results/')
    for folder in folders:
        for image in os.listdir('static/Example Results/'+folder):
            images.append(folder+"/"+image)
    #print(images)
    return render_template("show_images.html", images=images)

@app.route('/showcase_testing/')
def show_testing_images():
    images = []
    # Load in all the images.
    folder = os.listdir('static/Test/')
    for image in folder:
        images.append(image)
    return render_template("show_testing_images.html", images=images)

@app.route('/showcase_training/')
def show_training_images():
    images = []
        # Load in all the images.
    folder = os.listdir('static/Train/')
    number_of_files = 0
    for image in folder:
        number_of_files += 1    
        images.append(image)
        if number_of_files%50 == 0:
            print(number_of_files)
        if number_of_files == 800: #if number of images equals the batch_size of 800, stop.
            break
    return render_template("show_training_images.html", images=images)

@app.route('/colorize/', methods=['GET', 'POST'])
def colorize_images():
    folder = "static/Test/"
    model_name='model'
    from utils import save_the_images, loadModel, prepare_accuracy_visualisation_images
    # Firstly, load in the model from previous training sessions
    model = loadModel(model_name)
    # Secondly, load in images to colorise (they only have the lightness channel)
    color_me = []
    color_me = prepare_accuracy_visualisation_images(color_me, folder)
    # Thirdly, Colorize the loaded images
    output = model.predict(color_me)
    output = output * 128 # Turn the -1 to 1 values into proper Lab values.

    # Finally, Save the colorized images
    save_the_images(output, color_me)

    ## Now load those saved images back up so they can be displayed:
    images = []
    # Load in all the images.
    folder = os.listdir('static/Result/')
    for image in folder:
        images.append(image)
    return render_template("show_predicted_images.html", images=images)

@app.route('/train/')
def train_model():
    # This will run main.py
    import main

    #Load the images after going through main.py
    folder = os.listdir('static/Result/')
    images = []
    for image in folder:
        images.append(image)
    return render_template("show_predicted_images.html", images=images)

    #return render_template("index.html")
if __name__ == "__main__":
    # note: due to keras version used, I have to disable debug and threaded to avoid
    # ModuleNotFoundError: No module named 'tensorflow_core.keras' error
    app.run(debug=False, threaded=False)#debug=True)#