# import keras
import tensorflow.keras as keras
import numpy as np

from image_processor import get_image_mnist

def predict():
    model = keras.models.load_model('model_mnist')
    pred = model.predict([[get_image_mnist()]])
    return np.argmax(pred)