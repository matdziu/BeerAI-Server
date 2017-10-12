import tensorflow as tf
from keras.applications import InceptionV3
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential

from Classifier import output_beer_labels


def init():
    bottleneck_features_extractor = InceptionV3(include_top=False, weights='imagenet', input_shape=(150, 150, 3))

    model = Sequential()
    model.add(Flatten(input_shape=(2048, 1)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(output_beer_labels), activation='softmax'))
    model.load_weights('beer_label_classifier_weights.h5')

    graph = tf.get_default_graph()

    return bottleneck_features_extractor, model, graph
