### THIS WILL BE THE UNDERLYING API.
from flask import Flask, render_template, url_for
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

@app.route('/colorize/')
def colorize_images(folder = "Test/", model_name='new_model'):
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
    return "It worked!"
#@app.route('/img/<path:filename>') 
#def send_file(filename): 
    #return send_from_directory(filename)

if __name__ == "__main__":
    app.run(debug=True)