# import keras
import tensorflow.keras as keras
import numpy as np

# import dataset
mnist = keras.datasets.mnist

# unpack it into train and test data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the data
# the data is arrays of numbers from 0-255 but we'd like to normalize, that is, scaling to 0-1 range
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

# adding the model
model = keras.models.Sequential()

# building the layers
# Flatten layer is used to flatten the input which is 28x28 matrix, so using Flatten it will be of form 784x1
model.add(keras.layers.Flatten())

# 2 hidden layers which each consist of 128 neurons
# The activation function is ReLU which is Rectified Linear Unit
model.add(keras.layers.Dense(128, activation=keras.activations.relu))
model.add(keras.layers.Dense(128, activation=keras.activations.relu))

# The final layer which is consist of 10 neurons, here 10 represents the classification of digits. 0-9
# Softmax is the activation function
model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

# compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fitting the model
# we here pass the images and their corresponding correct answers
# epochs is just a number that represents how many time we'll repeat the entire training
model.fit(x_train, y_train, epochs=4)

# predicting
predictions = model.predict([x_test])

# to see the result and test the first input
pred_num = np.argmax(predictions[0])
actual_num = y_test[0]

if pred_num == actual_num:
    print('Valid')

