# pandas library to import the dataset
import pandas as pd
# tensorflow for neural networks
import tensorflow as tf

# keras Sequential model, because we need a simple feed forward network
from tensorflow.keras.models import Sequential
# we need a Dense layer and Dropout
from tensorflow.keras.layers import Dense, Dropout

# to split the data into test and train parts
from sklearn.cross_validation import train_test_split
# to shuffle the dataset in order to remove any kind of order
from sklearn.utils import shuffle

# import dataset
df = pd.read_csv('./redwinequality.csv')

# shuffling the data frames by rows
df = shuffle(df)

# splitting the data into feature matrix and target vector
# feature matrix
X = df.iloc[:, :-1].values
# target vector
y = df.iloc[:, -1:].values

# normalizing the data so that everything will be between 0 and 1
X = tf.keras.utils.normalize(X, axis=1)

# train_test_split the data into test and train parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# neural network
# sequential model, basic feed forward MLP
model = Sequential()

# dense layer with 256 neurons, activation is ReLu, the shape is (number_of_columns_of_X, )
model.add(Dense(units=256, activation='relu', input_shape=(X_train.shape[1], )))
# adding Dropout to prevent over-fitting
model.add(Dropout(0.1))
# dense layer with 256 neurons, activation is ReLu
model.add(Dense(units=256, activation='relu'))
# adding Dropout to prevent over-fitting
model.add(Dropout(0.1))
# dense layer with 256 neurons, activation is ReLu
model.add(Dense(units=256, activation='relu'))
# dense layer with 10 neurons, because the range is classified in 10 classes
model.add(Dense(units=10, activation='softmax'))

# optimizer is the specialized gradient-descent algorithm, here Adam
# loss function is the overall cost function which assigns a cost to our models predictions
# network tries to minimize the loss/cost function
# metrics basically means what is the measurement for our performance
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# fitting
model.fit(X_train, y_train, epochs=250, batch_size=10)

# loss and accuracy based on validation data
loss, acc = model.evaluate(X_test, y_test)

print('Loss : '+str(loss)+'\nAcc : '+str(acc))

