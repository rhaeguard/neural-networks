# matplotlib for visualization
import matplotlib.pyplot as plt

# import Boston house prices dataset
from sklearn.datasets import load_boston
# import train_test_split
from sklearn.cross_validation import train_test_split
# import linear model
from sklearn.linear_model import LinearRegression

# load dataset
boston = load_boston()

# get feature matrix
features = boston.data

# get target array
target = boston.target

# splitting
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=1/4, random_state=42)

# initializing Linear Model
model = LinearRegression()

# fitting
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)

# visualize
plt.scatter(y_test, predictions)
plt.show()




