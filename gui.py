import os
#import flask library and Flask class
from flask import Flask, render_template, request, send_from_directory

gui = Flask(__name__, template_folder="template")

#the root of application in the server(absolute path of the working project)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#link to the domain
@gui.route("/")

def index():
    #return html hardcode
    return render_template("upload.html")

@gui.route("/upload", methods=["POST"])
def upload():
    #add images folder for storage to the root
    target = os.path.join(APP_ROOT, 'images/')
    #for debugging
    print(target)

    #if images folder does not exist
    if not os.path.isdir(target):
        #create such directory
        os.mkdir(target)
    else:
        print("Could not create upload directory: {}".format(target))
    print(request.files.getlist("file"))

    #loop through files that were submitted via form
    for upload in request.files.getlist("file"):
        #debugging
        print(upload.filename)
        #obtain filename from the list of objects
        filename = upload.filename
        #add file name to the folder that we want to store our img
        destination = "/".join([target, filename])
        print("Accept incoming file: ", filename)
        
        print(destination)
        #save file in the destination folder
        upload.save(destination)
    return render_template("complete.html", image_name=filename)

@gui.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    #run the gui
    gui.run(port=4555, debug=True)

    #something
