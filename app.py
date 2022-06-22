import base64
import io
import json
import os
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


app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

print('loading saved model...')
spiral_modal = tf.keras.models.load_model('saved_model/model_spiral')
# spiral_modal = ''


app.config["UPLOAD_FOLDER"] = "static/Images"


@app.route('/date', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin()
def get_time():
    x = datetime.datetime.now()
    # Returning an api for showing in  reactjs
    return {
        'Name':"geek", 
        "Age":"22",
        "Date":x, 
        "programming":"python"
        }


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

@app.route('/', methods=["GET", "POST"])
@app.route('/wave', methods=["GET", "POST"])
def via_wave():
    # if request.method == "POST":

    #     image = request.files['file']

    #     if image.filename == '':
    #         print("Image must have a file name")
    #         # return redirect(request.url)

    #         filename = secure_filename(image.filename)

    #         basedir = os.path.abspath(os.path.dirname(__file__))
    #         image.save(os.path.join(
    #             basedir, app.config["UPLOAD_FOLDER"], filename))
    #         img_path = os.path.join(basedir, app.config["UPLOAD_FOLDER"], filename)

    #         final = myModal(img_path)
    #     else : 
    #         return render_template('via_wave')

    # return render_template('output.html', text='Analysis of the Image', img_src=f"data:image/png;base64,{view_classify(image)}" )
    # return render_template('output.html', text=final )
    return render_template('via_wave.html' )

@app.route('/spiral', methods=["GET", "POST"])
def via_spiral():
    # if request.method == "POST":

    #     image = request.files['file']

    #     if image.filename == '':
    #         print("Image must have a file name")
    #         # return redirect(request.url)

    #         filename = secure_filename(image.filename)

    #         basedir = os.path.abspath(os.path.dirname(__file__))
    #         image.save(os.path.join(
    #             basedir, app.config["UPLOAD_FOLDER"], filename))
    #         img_path = os.path.join(basedir, app.config["UPLOAD_FOLDER"], filename)

    #         final = myModal(img_path)
    #     else : 
    #         return render_template('via_wave')

    # return render_template('output.html', text='Analysis of the Image', img_src=f"data:image/png;base64,{view_classify(image)}" )
    # return render_template('output.html', text=final )
    return render_template('via_spiral.html' )


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

    final = myModal(img_path)

    # return render_template('output.html', text='Analysis of the Image', img_src=f"data:image/png;base64,{view_classify(image)}" )
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

    final = myModal(img_path)

    # return render_template('output.html', text='Analysis of the Image', img_src=f"data:image/png;base64,{view_classify(image)}" )
    return render_template('via_spiral.html', result=final, filename=filename )

@app.route('/home', methods=["GET", "POST"])
def upload_image():
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

    final = myModal(img_path)

    # return render_template('output.html', text='Analysis of the Image', img_src=f"data:image/png;base64,{view_classify(image)}" )
    return render_template('output.html', text=final )


def myModal(img_path):
    # image = Image.open(image, target_size=(100,100))
    image=load_img(img_path ,target_size=(100,100))
    image = img_to_array(image)
    image = image/255.0
    prediction_image = np.array(image)
    prediction_image = np.expand_dims(image, axis=0)
    print('\r process image done.')

    print('Predict image...')
    prediction = spiral_modal.predict(prediction_image)
    print('\r Predict image  done.')
    value = np.argmax(prediction)
    move_name=mapper(value)
    print("Prediction is {}.".format(move_name))

    return move_name


@app.route('/test', methods=['GET', 'POST', 'OPTIONS'])
@cross_origin()
def test():
    print("---------------- test ----------------")
    
    req = request.get_json()
    # print(req['type'])
    print(req)



    # return { "status" : req}
    return {
        "msg" : 'hello from server'
    }









