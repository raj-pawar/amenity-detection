from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

from PIL import Image
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import json

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'retinanet_5_classes.h5'
with open('label_map.json','r') as f:
    label_map = json.load(f)

# Load your trained model
model = models.load_model(MODEL_PATH, backbone_name='resnet50')
model = models.convert_model(model)
#model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
	im = np.array(Image.open(img_path))
	imp = preprocess_image(im)
	imp, scale = resize_image(im)
	boxes, scores, labels = model.predict(np.expand_dims(imp, axis=0))
	predictions = []
	for box, score, label in zip(boxes[0], scores[0], labels[0]):
		if score < 0.5:
			break
		class_name = label_map[str(label)]
		predictions.append(class_name)	    

	return predictions


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
         # ImageNet Decode
                     # Convert to string
        if(len(preds) == 0):
        	return "Nothing Detected"
        else:
	        return str(preds)
    return None


if __name__ == '__main__':
    app.run(debug=True)

