# Author: Adrienne Rudolph
# Class: ENME 808
# HW1 Problem 1 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as rootMSE

#Read in the data
mlr05 = pd.read_excel("mlr05.xlsx")

#Prepare the data
X_varb = mlr05.iloc[:, 1:]  #X variables are the X2, X3, .. X6 predictors
Y_varb = mlr05.iloc[:, 0]   #Y variable is the sales data, X1

#Separate Training and Testing Sets
x_train = X_varb[:20] #training set are X_varb columns, first 20 rows
y_train = Y_varb[:20] #training set is 'sales data' column, first 20 rows

x_test = X_varb[20:] #test set are all columns, last 7 rows
y_test = Y_varb[20:] #test set is 'sales data' column, last 7 rows

#Select the linear regression model
dataModel = LinearRegression()

#Train the model
dataModel.fit(x_train, y_train)

#Make a prediction for the sales data
prediction = dataModel.predict(x_test)

#Evaluate for error in actual and prediction
rmse = rootMSE(y_test, prediction)
print(rmse)

#Reshape and print prediction and
prediction = prediction.reshape(-1,1) #Visually appealing column vector for printing purpose
print(prediction)

#Plot test and prediction, and show line of perfect prediction
plt.scatter(y_test, prediction, label='Predicted vs Actual')
plt.xlabel('Actual Sales Data')
plt.ylabel("Predictied Sales Data")
plt.title('Actual vs Predicted Sales Data Values')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='purple', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()

