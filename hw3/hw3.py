import numpy as np
import matplotlib.pyplot as plt

rad = 10 #inner radius
thk = 5  #thickness
sep = 5  #separation

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


# Call function and plot points

xsemi_red, xsemi_blue, ysemi_red, ysemi_blue = CreateSemiCircles(rad, thk, sep)
"""
plt.scatter(xsemi_red, ysemi_red, color='red')
plt.scatter(xsemi_blue, ysemi_blue, color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Double Semi-Circle Toy')
plt.axis('equal')
plt.show()
"""

## PLA
x = np.concatenate((xsemi_red, xsemi_blue))
y = np.concatenate((ysemi_red, ysemi_blue))
dataPts = np.column_stack((x, y))
label = np.concatenate([np.ones(1000), -1*np.ones(1000)])

w = np.zeros((3, 1))
x_vec = np.zeros((3, 1))
s_t = np.sign(np.dot(w.T, x_vec))
c = 0
while np.any(s_t != label) and c<len(label):
    x, y = dataPts[c]
    x_vec = np.array([1, x, y]).reshape(-1,1)
    s_t = np.sign(np.dot(w.T, x_vec))
    temp = label[c] * s_t
    if temp <= 1:
        w += (label[c] - s_t) * x_vec
    c += 1

w0 = w[0]
w1 = w[1]
w2 = w[2]
def output_PLA(z):
    return (-(w1/w2) * z) - (w0/w2)

z = np.linspace(-20,30)

## Linear Regression Method
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
def output_LR(f):
    return (-(w1_lin/w2_lin) * f) - (w0_lin/w2_lin)

f = np.linspace(-20,30)

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

