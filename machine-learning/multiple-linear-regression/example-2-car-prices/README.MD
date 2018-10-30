# Car Prices Prediction

The dataset is in csv form. It is the dataset of used car prices.
The columns are:
 - car
 - price
 - body 
 - mileage 
 - engV
 - engType
 - registration
 - year 
 - model 
 - drive
 
We're not gonna drop any column here, we'll use all the columns. Let's start.

## Building Multiple Linear Regression Model

Here are the steps:

- importing the libraries
- importing the dataset
- preprocessing
- splitting
- initializing and fitting the model
- visualization

### Importing the libraries

 - **pandas** library is used for getting the data from a csv file
 - **datatime** for getting the current year
 - **pyplot** for visualization
 - **train_test_split** for splitting the dataset into test and train
 - **LabelEncoder, OneHotEncoder** to encode the categorical variables
 - **LinearRegression** to build the linear model
 

```python
# pandas library to get the data out of csv file
import pandas as pd
# datetime to get the current year
import datetime
# PyPlot to visualize the results
import matplotlib.pyplot as plt

# train_test_split method from cross validation library to split the dataset into train and test parts
from sklearn.cross_validation import train_test_split
# label encoder and one hot encoder, to encode the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Linear Regression model
from sklearn.linear_model import LinearRegression
```

### Importing the dataset
```python
# import the dataset
# it contains non-utf-8 chars, so we gotta change encoding
df = pd.read_csv('./car_ad.csv', encoding="ISO-8859-1")
```

### Preprocessing
Here we remove any row which contains unusual characters
```python
# remove the rows with non-utf-8 chars which is in 'model' column
df = df[df['model'].str.isalnum()]
```
Removing the NaN rows. **inplace** parameter mentions that dropping affects the data frame 
```python
# remove NaN contained rows
df.dropna(inplace=True)
```
Split the data into feature and target parts
```python
# split the data into feature matrix and target vector
# save the PRICE column to a vector
TARGET_VECTOR = df['price'].values

# drop PRICE column from the data frame and save it as feature matrix
FEATURE_MATRIX = df.drop('price', axis=1)
```
Convert year column to ages by subtracting it from the current year.
Save the data frame as an array
```python
# convert year to age
# current year
year = datetime.datetime.now().year

# convert years to ages
FEATURE_MATRIX['year'] = year - FEATURE_MATRIX['year']

# DataFrame to multidimensional array
FEATURE_MATRIX = FEATURE_MATRIX.values
```
Encoding the categorical variables
```python
# ENCODING Categorical Variable
# the indices of categorical variables
category_indices = (0, 1, 4, 5, 7, 8)

# encode labels
for i in category_indices:
    label_encoder_X = LabelEncoder()
    FEATURE_MATRIX[:, i] = label_encoder_X.fit_transform(FEATURE_MATRIX[:, i])

# One hot encode the categorical values
one_hot_encoder = OneHotEncoder(categorical_features=category_indices)

# Encode and save it as an array
FEATURE_MATRIX = one_hot_encoder.fit_transform(FEATURE_MATRIX).toarray()
```
### Splitting
We gotta split the data into test and train. Our model will learn on train and we'll test it on test data
```python
# split into test and train
X_train, X_test, y_train, y_test = train_test_split(FEATURE_MATRIX, TARGET_VECTOR, test_size=1/4, random_state=42)
```
### Initializing and fitting the model
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