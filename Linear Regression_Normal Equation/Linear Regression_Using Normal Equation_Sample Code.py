import matplotlib.pyplot as plt
import numpy as np

#Let us assume x is the 1D feature and y is the target variable. Initiating the values of x and y with random numbers
x = np.array([40,51,44,52,76,12,48,53,98,42,23,65,46,2,98,87,72,96,90,100,29,3,41,99,20])
y = np.array([64,57,32,64,58,49,52,51,53,68,43,62,57,40,69,48,66,68,50,38,52,53,63,49,42])

#Converting x into a m x 2 matrixs - for 2 feature variables x0 = intercept and x1
X = np.ones((len(x),2))
X[:,1] = x

#Apply normal equation to derive the optimal values of co-efficients of x i.e. theta0 and theta1
X_trans = np.transpose(X)
Xt_X = np.dot(X_trans,X)
theta = np.dot(np.dot(np.linalg.inv(Xt_X), X_trans), y)

#Let us plot the x vs. y values in a 2D graph for visualizing the pattern
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (8,4))
axes.scatter(x,y)


# Y_Predicted values are derived using w0 and w1 values as offered by Normal Equation and finally applying Y_Pred = theta0 * x0 + theta1 * x1
y_pred = []
for i in range(len(y)):
    y_pred.append((theta[1]*x[i]+theta[0]))

#Now we need to derive the total cost i.e. squared errors
mse = 0
for i in range(len(y)):
    mse = mse + pow((y[i]-y_pred[i]),2)

#Once squared error is derived, we can derive mean of squared errors
mse = mse * (1/len(y))

print("Mean of Squared Error for the Linear Regression Line derived using Normal Equation is {0:.2f}".format(mse))


#Plot the predicted values on the same graph drawn for x vs. y
axes.plot(x,y_pred)
plt.tight_layout()