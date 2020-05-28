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

@app.route('/Test/')
def show_testing_images():
    images = []
    # Load in all the images.
    folder = os.listdir('static/Test/')
    for image in folder:
        images.append(image)
    return render_template("show_testing_images.html", images=images)

@app.route('/Train/')
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

@app.route('/imagine/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        filename = file.filename
        file.save(os.path.join('images', filename)) #check back on 'images'
        return redirect(url_for('uploaded_file',filename=filename))
    return '''
        <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
if __name__ == "__main__":
    app.run(debug=True)