import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import the training dataset into a pandas dataframe. The training data has 2 variables x = Distance in KMs driven by a driver, y = Profit in $
mydata = pd.read_csv("ex1data1.txt", header = None)


#Plot the training data in a 2D graph to visualize the data spread
fig, axes = plt.subplots(nrows=2, ncols =1, figsize = (8,4))
axes[0].scatter(mydata.iloc[:,0], mydata.iloc[:,1])
axes[0].set_xlabel("Distance in KMs")
axes[0].set_ylabel("Profit in $")
axes[0].set_title("Distance vs. Actual Profit")
plt.tight_layout()


#Initial the values of co-efficients theta0 = intercept and theta1 = x1 co-efficient
thetas = np.random.rand(2)

#thetas.reshape(1,2)

#Initialize the hyper-parameter learning rate alpha with 0.01. Please fine tune the learning rate for optimal performance againsta different training set
alpha = 0.01
cost_series = []
counter = 0

#Iteratively find out the optimal values of x co-efficients until the cost function convergers
while(True):
    counter = counter+1
    
    #Cost = MSE = Mean of squared errors for the training set
    cost = 0
    for i in range(len(mydata)):
        cost = cost + pow(((thetas[0]*mydata.iloc[i,0]+thetas[1])-mydata.iloc[i,1]),2)  #Predicted value of Y = theta0+theta1*x1. The difference w.r.t. to the actual value of Y is squared (to avoid negative values pulling down the costs)
    
    #Mean of Squared error is cost
    cost = cost/len(mydata)
    
    #Keeps track of cost valyes in each iteration
    cost_series.append(cost)

    #Using Gradient-descent principle, value of del_theta1 is calculated
    del_theta1 = 0
    for i in range(len(mydata)):
        del_theta1 = del_theta1 + (((thetas[0]*mydata.iloc[i,0]+thetas[1])-mydata.iloc[i,1]))
    del_theta1 = del_theta1*(2/len(mydata))
    
    #Using Gradient-descent principle, value of del_theta0 is calculated
    del_theta0 = 0
    for i in range(len(mydata)):
        del_theta0 = del_theta0 + (((thetas[0]*mydata.iloc[i,0]+thetas[1])-mydata.iloc[i,1])*mydata.iloc[i,0])
    del_theta0 = del_theta0 * (2/len(mydata))


    #theta values are update in line with G.D. formula
    thetas[0] = thetas[0] - alpha * del_theta0
    thetas[1] = thetas[1] - alpha * del_theta1
    
    #Iteration stops when the change in cost functional is <0.001 i.e. the cost function has converged
    if(counter>=2):
        if(abs(cost_series[counter-2] - cost_series[counter-1])<0.0001): #The value 0.0001 can be changed based on implementation needs
            break


#Plot how the cost values have changed over different iterations. This is important to analyze how the cost function has converged
axes[1].plot(cost_series)
axes[1].set_xlabel("# of iterations")
axes[1].set_ylabel("Cost")
axes[1].set_title("Cost vs. Iterations")
plt.tight_layout()

#Calculate the predicted valyes of as (w0 + w1*x1)
predict_y =[]
for i in range(len(mydata)):
    predict_y.append(thetas[0]*mydata.iloc[i,0]+thetas[1])
#Plot the linear prediction line in the first graph
axes[0].plot(mydata.iloc[:,0], predict_y)
