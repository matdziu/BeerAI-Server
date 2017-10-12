import tensorflow as tf
from keras.applications import InceptionV3
from keras.models import model_from_json

PROJECT_PATH = "/Users/mateuszdziubek/Desktop/BeerAI-Server"


def init():
    model_file = open(f'{PROJECT_PATH}/model.json', 'r')

    bottleneck_features_extractor = InceptionV3(include_top=False, weights='imagenet', input_shape=(150, 150, 3))
    model = model_from_json(model_file.read())
    model.load_weights('beer_label_classifier_weights.h5')

    graph = tf.get_default_graph()

    return bottleneck_features_extractor, model, graph
