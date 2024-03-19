# ENME 808
# HW 3, Question 4
# Adrienne Rudolph

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

#Import the train and test data
trainData = pd.read_csv('mnist_train_binary (2) (1) (1).csv')
testData = pd.read_csv('mnist_test_binary (1) (1) (1).CSV')

#Parse the training and test data
xTrain = np.array(trainData.drop(columns='label'))
xTest = np.array(testData.drop(columns='label'))
yTrain = np.where(trainData['label'] == 1, 1, -1)
yTest = np.where(testData['label'] == 1, 1, -1)

#Shrink data down to 2 components
pca = PCA(n_components=2)
xTrain = pca.fit_transform(xTrain)
xTest = pca.transform(xTest)

#If testing with 3rd degree polynomial:
n = 3
poly = PolynomialFeatures(n)
X_train_p3 = poly.fit_transform(xTrain)
X_test_p3 = poly.transform(xTest)

# Turn into a vector to be used in signal calculation
xTrain = np.concatenate((np.ones((xTrain.shape[0],1)),xTrain),1)
xTest = np.concatenate((np.ones((xTest.shape[0],1)),xTest),1)

def LinReg(xTrain, yTrain):
    X = xTrain
    Y = yTrain
    X_trans = X.T
    X_trans_X = np.dot(X_trans, X)
    X_inverse = np.linalg.inv(X_trans_X)
    X_t = np.dot(X_inverse, X_trans)
    w_linear = np.dot(X_t, Y)
    return w_linear

def predict(x):
    global w
    return np.sign(np.dot(x,w))

def update_weights(prediction,true_value,input_value):
    global w
    if prediction * true_value < 1:
        w += (true_value - prediction) * input_value

# Run linear regression to get initial weights - Pick whether you want x_train to be 
# regular regression, or regression with the 3rd poly
# Comment out the one you dont want to use
#x_train = xTrain
x_train = X_train_p3

# Same thing here, choose which test you want to use and comment out the other
#x_test = xTest
x_test = X_test_p3

w = LinReg(x_train, yTrain) #This changes depending on if regular regression, or reg + 3rd order poly

# Initialize the prediction, best weights, and best error
prediction = predict(x_train)
w_best = w
E_best = np.sum(predict(x_train) != yTrain)

# Run Pocket Algorithm to find best weights
for _ in range(10):
    for i in range(x_train.shape[0]):
        prediction = predict(x_train[i,:])
        update_weights(prediction,yTrain[i],x_train[i,:])
        new_acc = np.sum(predict(x_train) != yTrain)
        if new_acc < E_best:
            w_best = w
            E_best = new_acc
w = w_best

##########################################
#reg = 0 # Choose for Regular Regression
reg = 1 # Choose for 3RD POLY Regression
##########################################

### ONLY USE NEXT PORTION FOR REGRESSION + POCKET
# Use best weights to calculate the line of regression
if reg == 0:
    print(w)
    w0 = w[0]
    w1 = w[1]
    w2 = w[2]
    z = np.linspace(-1000,2000)
    zSlope = -w1 / w2
    zIntercept = -w0 / w2

    # Plot training set
    plt.figure(1)
    plt.plot(z, zSlope*z+zIntercept, color='purple', label='Linear Reg Line')
    plt.scatter(xTrain[yTrain==1][:,1],xTrain[yTrain==1][:,2], color='red', label='+1')
    plt.scatter(xTrain[yTrain!=1][:,1],xTrain[yTrain!=1][:,2], color='blue', label='-1')
    plt.title('Linear Regression + Pocket - TRAINING Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    # Plot testing set
    plt.figure(2)
    plt.plot(z, zSlope*z+zIntercept, color='purple', label='Linear Reg Line')
    plt.scatter(xTest[yTest==1][:,1],xTest[yTest==1][:,2], color='red' , label='+1')
    plt.scatter(xTest[yTest!=1][:,1],xTest[yTest!=1][:,2], color='blue', label='-1')
    plt.title('Linear Regression + Pocket - TESTING Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


# ### ONLY USE FOR REGRESSION + POCKET + 3RD POLY
# ## Only use for polynomial portion
if reg == 1:
    x1 = np.linspace(np.min(xTrain[:,1]), np.max(xTrain[:,1]),1000)
    x2 = np.linspace(np.min(xTrain[:,2]), np.max(xTrain[:,2]),1000)
    px, py = np.meshgrid(x1, x2)

    transformed_x = poly.transform(np.stack((px.flatten(), py.flatten()), 1))
    ty = predict(transformed_x)

    # THIS IS FOR POLYNOMIAL PLOTTING
    # Plot training set
    plt.figure(1)
    plt.contour(px, py, ty.reshape(x1.shape[0], -1),levels=[0])
    plt.scatter(xTrain[yTrain==1][:,1],xTrain[yTrain==1][:,2], color='red', label='+1')
    plt.scatter(xTrain[yTrain!=1][:,1],xTrain[yTrain!=1][:,2], color='blue', label='-1')
    plt.title('Linear Regression, Poly3 - Training Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

    # Plot testing set
    plt.figure(2)
    plt.contour(px, py, ty.reshape(x1.shape[0], -1),levels=[0])
    plt.scatter(xTest[yTest==1][:,1],xTest[yTest==1][:,2], color='red', label='+1')
    plt.scatter(xTest[yTest!=1][:,1],xTest[yTest!=1][:,2], color='blue', label='-1')
    plt.title('Linear Regression, Poly3 - Testing Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

#Linear Regression Error Calculations
E_test_in = np.sum(predict(x_test) != yTest) / yTest.shape[0]
E_test_out = E_test_in + np.sqrt((1/(2*x_test.shape[0]))*np.log(2/0.05))
E_train_in = np.sum(predict(x_train) != yTrain) / yTrain.shape[0]
E_train_out = E_train_in + np.sqrt((8/x_train.shape[0]*np.log(4*(np.power(2*x_train.shape[0],3)+1)/0.05)))

if reg == 0:
    print("E_test_in for linear regression equals ", E_test_in)
    print("E_train_in for linear regression equals ", E_train_in)

    print("E_test_out for linear regression equals ", E_test_out)
    print("E_train_out for linear regression equals ", E_train_out)

if reg == 1:
    print("E_test_in for 3rd-order polynomial equals ", E_test_in)
    print("E_train_in for 3rd-order polynomial equals ", E_train_in)

    print("E_test_out for 3rd-order polynomial equals ", E_test_out)
    print("E_train_out for 3rd-order polynomial equals ", E_train_out)