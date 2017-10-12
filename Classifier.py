import numpy as np

PROJECT_PATH = "/Users/mateuszdziubek/Desktop/BeerAI-Server"

labels_file = open(f"{PROJECT_PATH}/labels.txt", "r")
output_beer_labels = labels_file.read().split('\n')


def classify(prediction_encoded):
    return output_beer_labels[np.argmax(prediction_encoded)]
