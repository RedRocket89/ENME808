import numpy as np
import matplotlib.pyplot as plt

rad = 10 #inner radius
thk = 5  #thickness
sep = np.arange(0.2, 5.1, 0.2)
# sep = 0.2


def CreateSemiCircles(rad, thk, sep):
    r_red = np.random.uniform(rad, rad+thk, 1000) #radius for the red circle
    theta1 = np.random.uniform(0, np.pi, 1000) #angle for the red circle

    r_blue = np.random.uniform(rad, rad+thk, 1000) #radius for the blue circle
    theta2 = np.random.uniform(np.pi, 2*np.pi, 1000) #angle for the blue circle

    #convert from radial coordinate system to cartesian coordinate system
    xsemi_red = r_red * np.cos(theta1)
    xsemi_blue = (r_blue * np.cos(theta2)) + (rad + (thk/2))

    ysemi_red = r_red * np.sin(theta1)
    ysemi_blue = (r_blue * np.sin(theta2)) - (sep)
    return xsemi_red, xsemi_blue, ysemi_red, ysemi_blue



def PerceptronAlg(xsemi_red, xsemi_blue, ysemi_red, ysemi_blue):
    ## PLA
    global w
    x = np.concatenate((xsemi_red, xsemi_blue))
    y = np.concatenate((ysemi_red, ysemi_blue))
    dataPts = np.column_stack((x, y))
    label = np.concatenate([np.ones(1000), -1*np.ones(1000)])

    w = np.zeros(3)

    def predict(x):
        return np.sign(np.dot(x,w))
    
    def updateWeights(prediction,true_value,input_value):
        global w
        if prediction * true_value <= 1:
            w += (true_value - prediction)*input_value

    x_vec = np.concatenate((np.ones((dataPts.shape[0],1)),dataPts),1)
    iteration = 0
    while np.any(predict(x_vec) != label):
        pt_index = iteration % x_vec.shape[0]
        pred = predict(x_vec[pt_index,:])
        updateWeights(pred,label[pt_index],x_vec[pt_index,:])
        iteration += 1  

    w0 = w[0]
    w1 = w[1]
    w2 = w[2]
    return w0, w1, w2, iteration, dataPts, label



def output_PLA(z, w0, w1, w2):
    eqn_pla = (-(w1/w2) * z) - (w0/w2)
    return eqn_pla



## Linear Regression Method
def LinReg(dataPts, label):
    ones = np.ones((dataPts.shape[0], 1))
    X = np.hstack((ones, dataPts))
    Y = label.reshape(-1,1)

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


f = np.linspace(-20,30)
z = np.linspace(-20,30)

def plotting(rad, thk, sep, f, z)
    xsemi_red, xsemi_blue, ysemi_red, ysemi_blue = CreateSemiCircles(rad, thk, sep)
    plt.scatter(xsemi_red, ysemi_red, color='red', label='+1 Data')
    plt.scatter(xsemi_blue, ysemi_blue, color='blue', label='-1 Data')
    plt.plot(z, output_PLA(z), color='green', label='PLA Hypothesis')
    plt.plot(f, output_LR(f), color = 'purple', label='Linear Reg')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Double Semi-Circle Toy - Data Separation')
    plt.legend()
    plt.xlim(-20,30)
    plt.ylim(-22,17)
    plt.show()

#######################################
# Problem 3.2

for_what = []
for a in range(len(sep)):
    xsemi_red, xsemi_blue, ysemi_red, ysemi_blue = CreateSemiCircles(rad, thk, sep[a])
    w0, w1, w2, total_iters, dataPts, label = PerceptronAlg(xsemi_red, xsemi_blue, ysemi_red, ysemi_blue)
    for_what.append(total_iters)
    print(a)

plt.plot(for_what)
plt.show()
    