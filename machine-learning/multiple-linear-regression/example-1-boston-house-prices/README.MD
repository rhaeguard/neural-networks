# Boston House Prices

In this example we'll go through very simple Multiple Linear Regression process.
The dataset we're going to use is called Boston House Prices dataset.

Let's examine the dataset a little bit. **Price** column normally doesn't reside
in the `data` part of the dataset. It is mentioned as `target`. But here they are stacked together.

![](../../../images/boston.png)

So basically, we assume that all the columns shown in the table above, somehow affect the last column in a linear fashion.
Therefore, our aim is to find that relationship and predict the future house prices based on their
features.

## Building Multiple Linear Regression Model

Here are the steps:

- importing the libraries
- importing the dataset
- preprocessing
- splitting
- initializing and fitting the model
- visualization

### Importing The Libraries

In this example, we'll need PyPlot to visualize the result. `load_boston()` method, loads house prices
data. `train_test_split` method enables us to split the data into train and test parts. In the 
last one we import the linear model which will actually be used to fit and predict. 

```python
# matplotlib for visualization
import matplotlib.pyplot as plt

# import Boston house prices dataset
from sklearn.datasets import load_boston
# import train_test_split
from sklearn.cross_validation import train_test_split
# import linear model
from sklearn.linear_model import LinearRegression

```

### Importing the dataset
In this dataset, we don't need to do any preprocessing, because it's ready-to-use.
```python
# load dataset
boston = load_boston()
```

### Preprocessing
For this example, by preprocessing, we mean to get the feature matrix and target vector. To do so:
```python
# get feature matrix
features = boston.data

# get target array
target = boston.target
```

### Splitting
As usual, we have to split the data into train and test sections. To make this partition much easier,
we'll use `train_test_split` method from `sklearn.cross_validation` library.

Here, `test_size` means, given the data(i.e features matrix and target vector), how much of the 
whole should be reserved for testing. For random state, it's stated that

`If random_state is None or np.random, then a randomly-initialized RandomState object is returned.If random_state is an integer, then it is used to seed a new RandomState object. If random_state is a RandomState object, then it is passed through.` 

```python
# splitting
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=1/4, random_state=42)
```

### Initializing and Fitting the Model

`predictions` is the array where the predictions for `X_test` input is saved.

```python
# initializing Linear Model
model = LinearRegression()

# fitting
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)
```

### Visualization

Since there are multiple features, we cannot efficiently show them on scatter plot or
any other plot. That's why we use different kind of way to depict the results.
We create the scatter plot of `y_test` and `predictions` where the former is the true value,
whereas the latter is the predicted numbers by our model. If the predictions are close the values
will tend to be closer to `y=x` line which passes through origin. 

```python
# visualize
plt.scatter(y_test, predictions)
plt.show()
```


