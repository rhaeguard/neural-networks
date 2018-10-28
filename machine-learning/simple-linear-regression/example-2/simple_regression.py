"""
Relation between LIKES and COMMENTS in Trending Videos

Simple Linear Regression
Dataset: US Trending Video Statistics
Aim: To see if there's a relation between like and comment counts in trending videos in US
"""
# numpy to do reshaping
import numpy as np
# pandas to import data from csv
import pandas as pd
# pyplot to visualize the output
import matplotlib.pyplot as plt

# scikit learn method to split the data into test and train parts
from sklearn.cross_validation import train_test_split
# LinearRegression model from scikit-learn to do simple linear regression
from sklearn.linear_model import LinearRegression

# import data from csv
df = pd.read_csv('./USvideos.csv')

# get the likes, that is the feature vector and convert it into ndarray
likes = df.iloc[:, 8].values
# get the comments, that is the target vector and convert it into ndarray
comments = df.iloc[:, 10].values

# the arrays are in 1D form, we have to reshape them to be flat and (n, 1) form
# reshape the likes
likes = np.reshape(likes, newshape=(len(likes), 1))
# reshape the comments
comments = np.reshape(comments, newshape=(len(comments), 1))

# splitting the data using train_test_split
# test_size is amount of the data that is going to be allocated to testing
likes_Train, likes_Test, comments_Train, comments_Test = train_test_split(likes, comments, test_size=0.25, random_state=0)

# initialize the model
model = LinearRegression()

# fitting the model
model.fit(likes_Train, comments_Train)

# predicted comment counts based on test like counts
comments_Predicted = model.predict(likes_Test)

# plotting
# scatter plot of test data
plt.scatter(likes_Test, comments_Test)
# plot test and predictions couples as a line
# aim is to visualize how much our prediction deviates from the original
plt.plot(likes_Test, comments_Predicted, color='red')
# title
plt.title('Relation between LIKES and COMMENTS in Trending Videos')
# x-axis labeling
plt.xlabel('Likes')
# y-axis labeling
plt.ylabel('Comments')

# show the plot
plt.show()
