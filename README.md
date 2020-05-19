# KeepingUpAppearances

We decided to tackle image colorization.
We used jupyter notebook for the IDE and Keras, skimage and numpy for Python module we wanted to work with.
Flask (or flask_restful, we'll see as the API develops) was used for API and GUI.

Alpha version of our baseline can be seen in the alpha_version_notebook.ipynb
Baseline can be seen in baseline.ipynb

Alpha version works with one photo at a time. Epochs value works best around 1000. Performance is bad with cold color pallet.

Baseline takes an entire folder of data, takes about 5% of it for testing and evaluating, uses the rest for training.

Check out the 'results' folder for when it ran across 10 images for about half an hour on a laptop.

## Requirements

**Required Libraries**:
keras
skimage
numpy
flask/flask_restful

To install
> pip install keras skimage flask_restful

**Dataset used**:
[https://www.floydhub.com/emilwallner/datasets/colornet](https://www.floydhub.com/emilwallner/datasets/colornet)
This dataset contains almost 10 thousand photos, all 256x256, that I used for the training session.

*Alpha version was based on*:
[https://www.freecodecamp.org/news/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d/](https://www.freecodecamp.org/news/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d/)
