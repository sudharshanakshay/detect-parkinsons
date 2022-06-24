import base64
import io
import json
import os
from posixpath import split
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask, request, make_response, Response, jsonify, redirect, render_template
import numpy as np
import datetime
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_cors import CORS, cross_origin
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from werkzeug.utils import secure_filename
import pickle
import re


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

print('loading saved model...')
spiral_modal = tf.keras.models.load_model('saved_model/model_spiral')
wave_modal = tf.keras.models.load_model('saved_model/model_wave')
filename = './saved_model/finalized_model.sav'
voice_model = pickle.load(open(filename, 'rb'))
# spiral_modal = ''


app.config["UPLOAD_FOLDER"] = "static/Images"


def wave_modal_func(img_path):
    image = load_img(img_path, target_size=(100, 100))
    image = img_to_array(image)
    image = image/255.0
    prediction_image = np.array(image)
    prediction_image = np.expand_dims(image, axis=0)
    print('\rprocess image done.')
    print('Predict image...')
    prediction = wave_modal.predict(prediction_image)
    print('\r Predict image  done.')
    value = np.argmax(prediction)

    pd_result = mapper(value)
    print("Prediction is {}.".format(pd_result))

    return pd_result


def spiral_modal_func(img_path):
    image = load_img(img_path, target_size=(100, 100))
    image = img_to_array(image)
    image = image/255.0
    prediction_image = np.array(image)
    prediction_image = np.expand_dims(image, axis=0)
    print('\rprocess image done.')
    print('Predict image...')
    prediction = spiral_modal.predict(prediction_image)
    print('\r Predict image  done.')
    value = np.argmax(prediction)

    # print(value)
    pd_result = mapper(value)
    print("Prediction is {}.".format(pd_result))

    return pd_result


def voice_modal_func(text):
    # image=load_img(img_path ,target_size=(100,100))
    # image = img_to_array(image)
    # image = image/255.0
    # prediction_image = np.array(image)
    # prediction_image = np.expand_dims(image, axis=0)
    # print('\rprocess image done.')
    # print('Predict image...')
    prediction = voice_model.predict(text)
    print('\r Predict   done.')
    # value = np.argmax(prediction)

    # print(value)
    # pd_result = mapper(value)
    print("Prediction is {}.".format(prediction))

    if (prediction[0] == 0):
        return "Healthy"
    else:
        return "Parkinsons" 


def mapper(value):
    Name = ['Parkinsons', 'Healthy']
    if value:
        return Name[1]
    else : 
        return Name[0]
	


@app.route('/', methods=["GET", "POST"])
@app.route('/wave', methods=["GET", "POST"])
def via_wave():
    return render_template('via_wave.html' )

@app.route('/spiral', methods=["GET", "POST"])
def via_spiral():
    return render_template('via_spiral.html' )

@app.route('/voice', methods=["GET", "POST"])
def via_voice():
    return render_template('via_voice.html' )

@app.route('/diagnose_wave', methods=["GET", "POST"])
def diagnose_wave():
    if request.method == "POST":

        image = request.files['file']

        if image.filename == '':
            print("Image must have a file name")
            return redirect(request.url)

        filename = secure_filename(image.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        image.save(os.path.join(
            basedir, app.config["UPLOAD_FOLDER"], filename))
        img_path = os.path.join(basedir, app.config["UPLOAD_FOLDER"], filename)

    final = wave_modal_func(img_path)

    return render_template('via_wave.html', result=final, filename=filename )


@app.route('/diagnose_spiral', methods=["GET", "POST"])
def diagnose_spiral():
    if request.method == "POST":

        image = request.files['file']

        if image.filename == '':
            print("Image must have a file name")
            return redirect(request.url)

        filename = secure_filename(image.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        image.save(os.path.join(
            basedir, app.config["UPLOAD_FOLDER"], filename))
        img_path = os.path.join(basedir, app.config["UPLOAD_FOLDER"], filename)

    final = spiral_modal_func(img_path)

    return render_template('via_spiral.html', result=final, filename=filename )

@app.route('/diagnose_voice', methods=["GET", "POST"])
def diagnose_voice():
    text = request.form['voice']

    print("================================")
    split_text = re.split(',', text)
    print(split_text)
    print("================================")

    not_final = list()
    final = list()


    for x in split_text:
        not_final.append(float(x))

    final.append(not_final)
    # if request.method == "POST":

        # text = request.files['voice']

        # print(text)

        # if image.filename == '':
        #     print("Image must have a file name")
        #     return redirect(request.url)

        # filename = secure_filename(image.filename)

        # basedir = os.path.abspath(os.path.dirname(__file__))
        # image.save(os.path.join(
        #     basedir, app.config["UPLOAD_FOLDER"], filename))
        # img_path = os.path.join(basedir, app.config["UPLOAD_FOLDER"], filename)

    result = voice_modal_func(final)

    return render_template('via_voice.html', result=result)





