import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


#Import the train and test data
trainData = pd.read_csv('mnist_train_binary (2) (1) (1).csv')
testData = pd.read_csv('mnist_test_binary (1) (1) (1).CSV')

X_train = np.array(trainData.drop(columns='label'))
Y_train = np.array(trainData['label'])

X_test = np.array(testData.drop(columns='label'))
Y_test = np.array(testData['label'])

pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

###############################################################

def LinReg(X_train, Y_train):
    ones = np.ones((X_train.shape[0], 1))
    X = np.hstack((ones, X_train))
    Y = Y_train.reshape(-1,1)

    X_trans = X.T
    X_trans_X = np.dot(X_trans, X)
    X_inverse = np.linalg.inv(X_trans_X)
    X_t = np.dot(X_inverse, X_trans)
    w_linear = np.dot(X_t, Y)
    print(w_linear)

    w0_lin = w_linear[0]
    w1_lin = w_linear[1]
    w2_lin = w_linear[2]
    return w0_lin, w1_lin, w2_lin


def output_LR(f, w0_lin, w1_lin, w2_lin):
    eqn_lr = (-(w1_lin/w2_lin) * f) - (w0_lin/w2_lin)
    return eqn_lr


def predict(x, w):
        return np.sign(np.dot(x,w))

    
def updateWeights(prediction,true_value,input_value, w):
    if prediction != true_value:
        w += (true_value - prediction)*input_value
    return w

#run LinReg first
def perceptron(X_train, label, w):       
    x_vec = np.concatenate((np.ones((X_train.shape[0],1)),X_train),1)
    iteration = 0
    for _ in range(5 * x_vec.shape[0]):
        pt_index = iteration % x_vec.shape[0]
        pred = predict(x_vec[pt_index,:], w)
        old_w = w
        old_accuracy = np.sum(predict(x_vec, old_w) != label)
        w = updateWeights(pred,label[pt_index],x_vec[pt_index,:], old_w)
        iteration += 1 
        print(iteration)
        new_accuracy = np.sum(predict(x_vec, old_w) != label)
        if old_accuracy < new_accuracy:
             w = old_w

    w0 = w[0]
    w1 = w[1]
    w2 = w[2]
    return w0, w1, w2, iteration, dataPts, label

def output_PLA(z, w0, w1, w2):
    eqn_pla = (-(w1/w2) * z) - (w0/w2)
    return eqn_pla

########################################################
w0_lin, w1_lin, w2_lin = LinReg(X_train, Y_train)
w = [w0_lin, w1_lin, w2_lin]

label = Y_train
w0, w1, w2, iteration, dataPts, label = perceptron(X_train, label, w)

z = np.linspace(-1000, 1000)
eqn_pla = output_PLA(z, w0, w1, w2)

plt.scatter(dataPts[:,0], color='red')
plt.scatter(dataPts[:,1], color='blue')
plt.plot(z, output_PLA(z, w0, w1, w2), color='purple')
plt.show()