# Simple Linear Regression Example

In this example, we try to predict the salary based on the years of experience.

First of all, we gotta import the necessary libraries and methods

```python
# pandas library is used to import the dataset from a csv file
import pandas as pd
# Linear Regression model to make a simple linear regression
from sklearn.linear_model import LinearRegression
# scikit-learn method for splitting the data into test and train parts
from sklearn.cross_validation import train_test_split
# Pyplot for visualization
import matplotlib.pyplot as plt
```

If those aren't available, to install them you should fire up the following commands:

```
pip install pandas
pip install scikit-learn
pip install matplotlib
```

The next step is to import the data from csv file

```python
df = pd.read_csv('./Salary_Data.csv')
```

After importing the dataset, we have to divide it into feature dataset and the target.
 - Feature is the independent variable in linear regression model. For example, here experience is the feature because based on experience, we'd like to predict the salary
 - Target is the value that we want to predict, that is salary in our case
 
```python
# find the feature vector that is independent variable
X = df.iloc[:, :-1].values
# find the target vector that is dependent variable
y = df.iloc[:, 1:].values
```

Using Scikit-learn library, we can divide the dataset into test and train data.
Train data is used to train our model and adjust its weights. Test data is the data that our model isn't aware of. It is used to 
measure its performance

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
``` 

Initializing the model. Fitting is the main part where the actual fitting occurs. Prediction part is the testing section of our model.

```python
# create the linear regression model
model = LinearRegression()

# model fitting
model.fit(X_train, y_train)

# predictions of test values
pred = model.predict(X_test)
```

For visualization, we can use Pyplot from matplotlib

```python
# Plotting
# scatter plot of test data to see where they are located
plt.scatter(X_test, y_test)
# plot test and predictions couples as a line
# aim is to visualize how much our prediction deviates from the original
plt.plot(X_test, pred, color='red')

# show the plot
plt.show()
```
