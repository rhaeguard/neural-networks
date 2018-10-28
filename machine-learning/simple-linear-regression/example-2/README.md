# Simple Linear Regression

In this example, we'll try to find the linear relationship between the number of likes and 
comments in trending Youtube videos and attempt to predict the comment count based on the likes the video has

Let's divide this process into several parts

- importing libraries
- importing the dataset
- data pre-processing (we're preparing the data for the model)
- data splitting into test and train
- fitting the model
- visualizing

### Importing libraries

```python
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
```

If those aren't available, to install them you should fire up the following commands:

```
pip install numpy
pip install pandas
pip install scikit-learn
pip install matplotlib
```

### Importing the dataset

```python
df = pd.read_csv('./USvideos.csv')
```

### Data Pre-processing

Selecting the likes and comments from the dataset

```python
# get the likes, that is the feature vector and convert it into ndarray
likes = df.iloc[:, 8].values
# get the comments, that is the target vector and convert it into ndarray
comments = df.iloc[:, 10].values
```

Reshaping the array, because it is 1D (`1 x n`), we need to make it `n x 1`
```python
# the arrays are in 1D form, we have to reshape them to be flat and (n, 1) form
# reshape the likes
likes = np.reshape(likes, newshape=(len(likes), 1))
# reshape the comments
comments = np.reshape(comments, newshape=(len(comments), 1))
```

### Splitting
We split the data into training and testing parts
```python
likes_Train, likes_Test, comments_Train, comments_Test = train_test_split(likes, comments, test_size=0.25, random_state=0)
```

### Fitting

```python
# initialize the model
model = LinearRegression()

# fitting the model
model.fit(likes_Train, comments_Train)

# predicted comment counts based on test like counts
comments_Predicted = model.predict(likes_Test)
```

### Visualizing
```python
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
```