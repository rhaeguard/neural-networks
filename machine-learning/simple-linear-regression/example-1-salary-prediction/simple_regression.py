"""
Simple Linear Regression
Dataset: Years of Experience and Salary dataset
Aim: To predict the salary based on the given years of experience
"""
# pandas library is used to import the dataset from a csv file
import pandas as pd
# Linear Regression model to make a simple linear regression
from sklearn.linear_model import LinearRegression
# scikit-learn method for splitting the data into test and train parts
from sklearn.cross_validation import train_test_split
# Pyplot for visualization
import matplotlib.pyplot as plt

# import the data from csv
df = pd.read_csv('./Salary_Data.csv')

# find the feature vector that is independent variable
X = df.iloc[:, :-1].values
# find the target vector that is dependent variable
y = df.iloc[:, 1:].values

# splitting into 2 parts : test and train
# test size here is that how much of the given dataset should be partitioned as test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# create the linear regression model
model = LinearRegression()

# model fitting
model.fit(X_train, y_train)

# predictions of test values
pred = model.predict(X_test)

# Plotting
# scatter plot of test data to see where they are located
plt.scatter(X_test, y_test)
# plot test and predictions couples as a line
# aim is to visualize how much our prediction deviates from the original
plt.plot(X_test, pred, color='red')

# show the plot
plt.show()


