import io
import os
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


import matplotlib.pyplot as plt
import tensorflow as tf


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
app = Flask(__name__)


@app.route('/print-plot')
def plot_png():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = np.random.rand(100)
    ys = np.random.rand(100)
    axis.plot(xs, ys)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def mapper(value):
	dir_sp_train = './archive/spiral/testing'
	Name = []
	for file in os.listdir(dir_sp_train):
		Name += [file]
	print(Name)
	print(len(Name))
	
	N = []
	for i in range(len(Name)):
		N += [i]
	
	reverse_mapping = dict(zip(N, Name)) 
	return reverse_mapping[value]


def myModal():
    spiral_modal = tf.keras.models.load_model('saved_model/model_spiral')

    image = load_img("./archive/spiral/testing/healthy/V55HE12.png", target_size=(100,100))
    image = img_to_array(image)
    image = image/255.0
    prediction_image = np.array(image)
    prediction_image = np.expand_dims(image, axis=0)

    prediction = spiral_modal.predict(prediction_image)
    value = np.argmax(prediction)
    move_name=mapper(value)
    print("Prediction is {}.".format(move_name))


myModal()
