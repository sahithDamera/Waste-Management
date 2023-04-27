import json
import numpy
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# Load the model
with open('model_prediction.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a graph for the model
graph = tf.get_default_graph()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    img_file = request.files['image']
    img = load_img(img_file, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    global graph
    with graph.as_default():
        output = model.predict(x)

    if output > 0.5:
        prediction = 'Recycle Waste'
    else:
        prediction = 'Organic Waste'

    return render_template('home.html', prediction_text=f'The image is classified as {prediction}.')


if __name__ == '__main__':
    app.run(debug=True)
