from flask import Flask, render_template, request, redirect, url_for
from flask import jsonify
from keras.models import load_model
from matplotlib import pyplot
from PIL import Image
from pylab import *
import os
import numpy as np
import cv2
import json
import scipy.misc as image
from werkzeug.utils import secure_filename
from PIL import Image
import base64
import re
import cStringIO
import random
import tensorflow as tf

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/dropdown')
def dropdown():
    return render_template('dropdown.html')

@app.route('/navbar')
def navbar():
    return render_template('navbar.html')

@app.route('/button')
def button():
    return render_template('button.html')

@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    model = load_model('./model/my_model_flow.h5')
    im = image.imread('/Users/savinaynarendra/Downloads/test1.png')
    im = image.imresize(im, (150,150))
    img = np.expand_dims(im, axis=0)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/save', methods=['POST'])
def get_image():
    model = load_model('./model/model.h5')
    image_b64 = request.values['imageBase64']
    clss = request.values['class']
    random_number = random.randint(0, 100000)
    file_name = clss + str(random_number)
    image_data = re.sub('^data:image/.+;base64,', '', image_b64).decode('base64')
    image_PIL = Image.open(cStringIO.StringIO(image_data))
    image_np = np.array(image_PIL)
    # print image_np
    im = image.imresize(image_np, (150,150))
    img = np.expand_dims(im, axis=0)
    arr = model.predict(img, batch_size=16, verbose=0)
    arr = arr.tolist()
    return jsonify(arr)
    
    # return "Success"

if __name__ == "__main__":
    # model = load_model('./model/my_model_flow_trained.h5')
    app.run(debug=False, host='0.0.0.0')