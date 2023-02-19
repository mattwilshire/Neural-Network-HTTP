import os
import numpy as np
import io
from PIL import Image

from tensorflow.keras.models import load_model
model = load_model('best_network.h5')

from flask import Flask, send_file, abort, request
import os.path
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "Index Page"

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    data = image.read()
    result = model.predict(preprocess_image(data))
    return result[0][0]

@app.route('/predict_multi', methods=['POST'])
def multi_files():
    files = request.files.getlist("image")
    
    dataset = []
    for file in files:
        dataset.append(preprocess_image(file.read()))

    input_tensor = np.concatenate(dataset, axis=0)
    results = model.predict(input_tensor)

    output = ""
    for result in results:
        output += str(format(result[0], '.12f')) + " "
        
    return outputs

def preprocess_image(file):
    img = Image.open(io.BytesIO(file))
    img = img.resize((224, 224)) # assuming the model expects 224x224 images
    img = np.array(img)
    img = np.expand_dims(img, axis=0) # add batch dimension
    return img


if __name__ == '__main__':
    app.run(debug=False)