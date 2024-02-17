import numpy as np
import matplotlib.pyplot as plt


#pick a random function x(t), y(t)
def targetFunc(x):
    return 3*x + (1/2)

dataPts = np.random.uniform(-10, 10, size=(10100, 2))

trainPts = dataPts[:100, :]

testPts = dataPts[100:, :]

train_feature = trainPts[:, 0]  #x data for training set
train_label = trainPts[:, 1]    #y data for training set

test_feature = testPts[:, 0]   #x data for test set
test_label = testPts[:, 1]    #y data for test set

#compute s(t) = w^Tx
w = np.zeros((3, 1))
n = 0.0001
y_train_label = []

for i in range(0, 10):

    for t in range(len(trainPts)):
        y_actual = trainPts[t,1] - targetFunc(trainPts[t,0])
        if y_actual > 0:
            y_actual = 1
        else:
            y_actual = -1
        if i == 0:
            y_train_label.append(y_actual)
        x_vec = np.array([1, trainPts[t,0], trainPts[t,1]]).reshape(-1,1)
        
        s_t = np.sign(np.dot(w.T, x_vec))
        
        temp = y_actual * s_t
        
        if temp <= 1:
            w += n * (y_actual - s_t) * x_vec        
w = w

w0 = w[0]
w1 = w[1]
w2 = w[2]
def output(z):
    return (-(w1/w2) * z) - (w0/w2)

#Apply best weights to test set
predictions = []
actual = []
for a in range(len(testPts)):
    y_actual_test = testPts[a,1] - targetFunc(testPts[a,0])
    if y_actual_test > 0:
        y_actual_test = 1
    else:
        y_actual_test = -1
    actual.append(y_actual_test)
    x_vec = np.array([1, test_feature[a], test_label[a]]).reshape(-1,1)

    s_t = np.sign(np.dot(w.T, x_vec))
    predictions.append(s_t)
y_test_label = actual
predictions = np.array(predictions).flatten()
actual = np.array(actual).flatten()
print(np.sum(predictions==actual)/predictions.shape[0])

y_train_label = np.array(y_train_label).flatten().astype(int)
y_test_label = np.array(y_test_label).flatten().astype(int)

# Plot the training data set

z = np.linspace(-10,10)

plt.scatter(train_feature[y_train_label==1], train_label[y_train_label==1], color='purple',label='+1')
plt.scatter(train_feature[y_train_label==-1], train_label[y_train_label==-1],color='red',label='-1')
plt.plot(z, targetFunc(z), color='blue', label='Target Function')
plt.plot(z, output(z), color='green', label='Final Hypothesis')
plt.title('Target Function vs Hypothesis, n = 0.0001')
plt.xlabel('X - Train Feature Data')
plt.ylabel('Y - Train Label Data')
plt.legend()
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.show()



#plot the training data set, the target function f, and the final hypothesis f on the same figure
#report the error on the test set


 
