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

# import the dataset
# it contains non-utf-8 chars, so we gotta change encoding
df = pd.read_csv('./car_ad.csv', encoding="ISO-8859-1")

# remove the rows with non-utf-8 chars which is in 'model' column
df = df[df['model'].str.isalnum()]

# remove NaN contained rows
df.dropna(inplace=True)

# split the data into feature matrix and target vector
# save the PRICE column to a vector
TARGET_VECTOR = df['price'].values

# drop PRICE column from the data frame and save it as feature matrix
FEATURE_MATRIX = df.drop('price', axis=1)

# convert year to age
# current year
year = datetime.datetime.now().year

# convert years to ages
FEATURE_MATRIX['year'] = year - FEATURE_MATRIX['year']

# DataFrame to multidimensional array
FEATURE_MATRIX = FEATURE_MATRIX.values

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

# split into test and train
X_train, X_test, y_train, y_test = train_test_split(FEATURE_MATRIX, TARGET_VECTOR, test_size=1/4, random_state=42)

# initializing Linear Model
model = LinearRegression()

# fitting
model.fit(X_train, y_train)

# predict
predictions = model.predict(X_test)

# visualize
plt.scatter(y_test, predictions)
plt.show()



