# KeepingUpAppearances

We decided to tackle image colorization.
We used jupyter notebook at first, before moving to standard .py files.
Keras, skimage, flask, and numpy were Python module we wanted to work with.
Flask was used for API and GUI.

## Important

Before doing anything, a dataset needs to be in place.
It was too large to put on GitHub so it is linked below.

## Baseline is split across 3 different versions.

Alpha version of our baseline can be seen in the alpha_version_notebook.ipynb
Beta version of our baseline can be seen in baseline.ipynb
Final version of the baseline is split across main.py, model.py, colorize.py and utils.py

Alpha version works with one photo at a time. Epochs value works best around 1000.
Its performance is bad with cold color pallet or when a different image is used for testing and for predicting.

### Layout

To train the model, test its performance, and run some images through it: run main.py (it is esentially the training loop).
To run some of the functions independently, utils.py can be a great resource to check out.
Colorize.py is essentially what the user would see. It loads in the model and colorizes the selected photos.
To see the gui, run gui.py (requires template folder to work).

## Requirements

**Required Libraries**:

- keras
- skimage
- numpy
- flask

To install them:
> pip install keras skimage numpy flask

Os and json, which are also required, are usually included with python.

**Other setup**:

There are some default variables set around the program and its functions. Below is a list of those functions and their values. Users can change them freely in the code. Parsing arguments was part of the team missions, which we sadly had to drop but I feel like this is an ok alternative.

1. *load_images* loads images from *Train/* folder by default.
2. when training, *tensorboard* is only used when last parameter is False. Across 4 different machines I had varring results (based on installation, OS and so on) with getting callbacks to work, so it is *disabled by default*.
3. when *saving the model*, the *default name is "model"*. This means the model is saved as *model.json and its weights are saved as model.hdf5*.
4. When preparing accuracy visualisation images a.k.a testing the model, *the default folder to load the images from is "Test/"*.
5. When *images are* being saved, they are *saved under "Result/"*.

## References

**Dataset used**:
[https://www.floydhub.com/emilwallner/datasets/colornet](https://www.floydhub.com/emilwallner/datasets/colornet)
This dataset contains almost 10 thousand photos, all 256x256, that I used for the training session.

*Alpha version was based on*:
[https://www.freecodecamp.org/news/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d/](https://www.freecodecamp.org/news/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d/)
