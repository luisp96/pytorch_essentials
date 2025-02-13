from flask import Flask, request, jsonify
from torch_utils import transform_image, get_prediction
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# allowed file helper method
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST']) # API route for a POST method
def predict():
    # 1 load image
    # 2 image -> tensor
    # 3 prediction
    # 4 return json

    # error checking
    # check that the request method is POST
    if request.method == 'POST':
        file = request.files.get('file')
        # check that the file exists
        if file is None or file.filename == '':
            return jsonify({'error': 'No file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'Image file format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            data = {'prediction': prediction.item(),
                    'class_name': str(prediction.item())} # here if we were doing some other class names, then we would need to convert the resulting index from the prediction to a class label
            return jsonify(data)
        except:
            return jsonify({'error': 'Error during prediction'})

    # return jsonify({"result": 1})