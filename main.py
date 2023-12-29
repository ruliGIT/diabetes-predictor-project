import numpy as np
import pandas as pd
import sklearn 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV 


# Data Collection and Analysis
# PIMA Diabetes Dataset

# loading dataset to pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')

# printing first 5 rows of dataset
print(diabetes_dataset.head())

# numbers of rows and columnc in the dataset
print(diabetes_dataset.shape)

# statistical measures of the data
print(diabetes_dataset.describe())

# 1 = is diabetic, 0 = not diabetic
print(diabetes_dataset['Outcome'].value_counts())

print(diabetes_dataset.groupby('Outcome').mean())

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

# Data Standarization

scaler = StandardScaler()
scaler.fit(X)
standarized_data = scaler.transform(X)

print(standarized_data)

X = standarized_data
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

# Train/Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Training the Model
classifier = svm.SVC(kernel='linear')

# training the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# Model Evalution
# Accuracy Score

# accuracy score on the trainning data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy score of the training data: ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score of the test data: ', test_data_accuracy)

# Train the Support Vector Classifier with Hyper-parameter Tuning - GridsearchCV

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear']}  
  
grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, Y_train)

# Get the best parameters found by GridSearchCV
best_params = grid.best_params_

# Retrain the model using the best parameters
best_classifier = svm.SVC(**best_params)
best_classifier.fit(X_train, Y_train)

# Evaluate the accuracy of the model using the best parameters
best_predictions = best_classifier.predict(X_test)
best_accuracy = accuracy_score(best_predictions, Y_test)
print('Accuracy score using best parameters: ', best_accuracy)

# Making a Prediction System

input_data = (4,110,92,0,0,37.6,0.191,30)

# changing input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')

# Saving the Trained Model as binary file
import pickle

filename = 'trained_model.sav'
pickle.dump(classifier, open(filename, 'wb')) 

# read binary file into variable
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

input_data = (4,110,92,0,0,37.6,0.191,30)

# changing input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = loaded_model.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')