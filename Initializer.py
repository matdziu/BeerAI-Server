import tensorflow as tf
from keras.models import model_from_json

PROJECT_PATH = "/Users/mateuszdziubek/Desktop/BeerAI-Server"


def init():
    bottleneck_features_extractor_file = open(f'{PROJECT_PATH}/models/bottleneck_features_extractor.json', 'r')
    model_file = open(f'{PROJECT_PATH}/models/model.json', 'r')

    bottleneck_features_extractor = model_from_json(bottleneck_features_extractor_file.read())
    model = model_from_json(model_file.read())
    model.load_weights('beer_label_classifier_weights.h5')

    graph = tf.get_default_graph()

    return bottleneck_features_extractor, model, graph
