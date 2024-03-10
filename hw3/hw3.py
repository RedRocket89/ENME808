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
plt.scatter(xsemi_red, ysemi_red, color='red')
plt.scatter(xsemi_blue, ysemi_blue, color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Double Semi-Circle Toy')
plt.axis('equal')
plt.show()

## PLA
x = np.concatenate((xsemi_red, xsemi_blue))
y = np.concatenate((ysemi_red, ysemi_blue))
dataPts = np.column_stack((x, y))
#print(dataPts)






