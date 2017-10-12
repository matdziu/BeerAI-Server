import numpy as np

output_beer_labels = ['Harnaś', 'Kasztelan Niepasteryzowany', 'Kasztelan Pszeniczny', 'Miłosław Witbier (niebieski)',
                      'Perła Chmielowa', 'Perła Export', 'Somersby', 'Warka', 'Wojak', 'Żywiec Biały']


def classify(prediction_encoded):
    return output_beer_labels[np.argmax(prediction_encoded)]
