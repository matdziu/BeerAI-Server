import os

import numpy as np
from flask import Flask, request
from keras.preprocessing.image import load_img, img_to_array

from Classifier import classify
from Initializer import init

app = Flask(__name__)

bottleneck_features_extractor, model, graph = init()

to_predict_path = "/Users/mateuszdziubek/Desktop/BeerAI-Server/to_predict"
to_predict_name = "image_to_predict.jpg"


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    image_file = request.files['image_file']
    image_file.save(os.path.join(to_predict_path, to_predict_name))
    image = load_img(os.path.join(to_predict_path, to_predict_name), target_size=(150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255

    with graph.as_default():
        bottleneck_feature = bottleneck_features_extractor.predict(image)
        prediction_encoded = model.predict(bottleneck_feature)
        response = classify(prediction_encoded)
        return response


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
