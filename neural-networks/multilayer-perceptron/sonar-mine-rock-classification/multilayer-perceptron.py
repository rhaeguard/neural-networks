# import pandas to dataset importing from csv
import pandas as pd
# numpy for reshaping
import numpy as np
# keras for neural networks
import keras

# Label encoder to encode the categorical data
from sklearn.preprocessing import LabelEncoder
# shuffling the data so that there will remain no interdependency
from sklearn.utils import shuffle
# train_test_split to separate the data into test and train data
from sklearn.cross_validation import train_test_split

# basic Sequential feed forward model
from keras.models import Sequential
# dense layer
from keras.layers import Dense

# read the data from csv file into a DataFrame
# header is None because csv contains no header
df = pd.read_csv('./sonar-data.csv', header=None)
# it has unbalanced number of observations, so we don't include all of them
df = df.iloc[:97*2, :]
# shuffle it to get rid of the order
df = shuffle(df)

# feature matrix
X = df.iloc[:, :-1].values
# target vector
y = df.iloc[:, -1].values

# encoding the target vector
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)

# reshape it from (1, n) to (n, 1)
y = np.reshape(y, (y.shape[0], 1))

# split it into train and test data couples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# the actual neural networks
model = Sequential()
# a dense layer with 256 neurons
model.add(Dense(256, activation='relu', input_dim=X.shape[1]))
# 1 output because it is a binary classification
# if not the given one, then it is the other one
model.add(Dense(1, activation='sigmoid'))

# compiling
# loss is this time binary_crossentropy but categorical_crossentropy can also be used
model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# fitting the model
model.fit(X_train, y_train, epochs=100, batch_size=2)

# validation loss and validation accuracy
loss, acc = model.evaluate(X_test, y_test)





