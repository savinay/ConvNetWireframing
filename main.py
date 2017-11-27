from flask import Flask, render_template
from keras.models import load_model
from matplotlib import pyplot
from PIL import Image
from pylab import *
import os
import numpy as np
import cv2
import json
import numpy as np


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model('./model/my_model.h5')
    print "inside"
    im = cv2.imread("/Users/savinaynarendra/Downloads/canvas.png")
    im = cv2.cvtColor( im, cv2.COLOR_BGR2GRAY )
    print im
    resized = cv2.resize(im, (150,150), interpolation = cv2.INTER_AREA)
    print resized.shape
    resized = np.reshape(resized, ( 150, 150, 1))
    print resized.shape
    img = np.expand_dims(resized, axis=0)
    print img.shape
    print model
    print model.predict(img, batch_size=16, verbose=0)
    # print arr
    # return json.dumps(arr)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')