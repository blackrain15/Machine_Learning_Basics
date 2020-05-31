import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the input file having 3 columns - Area, # of Rooms and Cost in $1000
mydata = pd.read_csv("ex1data2.txt", header = None)

#Plot house area vs. cost
fig, axes = plt.subplots(nrows=5, ncols =1, figsize = (12,8))
axes[0].scatter(mydata.iloc[:,0], mydata.iloc[:,2], color = 'r')
axes[0].set_xlabel("Area")
axes[0].set_ylabel("Cost in $1000")
axes[0].set_title("Area vs. Cost")

#Plot # of rooms vs. cost
axes[1].scatter(mydata.iloc[:,1], mydata.iloc[:,2], color = 'b')
axes[1].set_xlabel("# of Bedrooms")
axes[1].set_ylabel("Cost in 1000$")
axes[1].set_title("# of Bedrooms vs. Cost")

plt.tight_layout()

#Defining a mx3 features matrix - where col1 = x0 = intercept =1, col2 = x1 = area, col3 = x2 = # of rooms
X = np.ones((len(mydata),3))

X[:,1] = (mydata.iloc[:,0])/(np.max(mydata.iloc[:,0])) #Feature scaling is done diving each value of house area with the maximum area value
X[:,2] = (mydata.iloc[:,1])/(np.max(mydata.iloc[:,1])) #Feature scaling is done diving each value of # of rooms with the maximum # of rooms

Y_norm = mydata.iloc[:,2]/10000 #Feature scaling is done on the target variable Y = cost of a house - dividing each value with 10,000 (i.e. covering cost to 10 millions)


#Random assignment of co-efficients (w0, w1, w2) to start the iteration
w = np.random.rand(3)

#Let us assign learning rate alpha to 0.01. Please modify the value to fine tune the model for a different dataset
alpha = 0.01

LR_cost_list = [] #Will store the total cost from a given assignment of w - against each of the iterations
counter = 0


#Iteratively look for optimal value of w - until the change in cost is <0.00001 i.e. the cost function has converged
while(True):
    
    # Y_Predicted = w0x0+w1x1+w2x2
    Y_Pred = w[0]*X[:,0]+w[1]*X[:,1]+w[2]*X[:,2]

    #Cost = MSE = Mean of sum of squared errors - where error = Y_Predicted - Y_Actual
    cost_list = pow((Y_Pred-Y_norm),2)
    cost = sum(cost_list)*(1/len(mydata))
    
    
    counter +=1
    
    #Keeps track of cost in every iteration
    LR_cost_list.append(cost)

    
    #Revisit the values of w0,w1,w2 as per gradient descent formula    
    diff = np.array(Y_Pred - Y_norm)
    delw0 = sum(diff * w[0])*(1/len(mydata))
    delw1 = sum(diff * w[1])*(1/len(mydata))
    delw2 = sum(diff * w[2])*(1/len(mydata))

    w[0] = w[0] - alpha*delw0
    w[1] = w[1] - alpha*delw1
    w[2] = w[2] - alpha*delw2

    #Check if the change in cost is below the threshold and the iteration should be stopped.
    if(counter>2):
        if(np.abs((LR_cost_list[counter-1]-LR_cost_list[counter-2]))<0.00001):  #Change the value 0.00001 if we need higher precision
            break

#Plot the cost function to check how the cost function has converged. This is important to ensure that Gradient descent is able to converge
axes[2].plot(LR_cost_list, color = 'g')
axes[2].set_xlabel("# of Iters")
axes[2].set_ylabel("Cost Func.")
axes[2].set_title("Cost vs. # of Iters.")

#Plot the actual distribution of costs as given in the training set in $10M
x = np.arange(len(mydata))
axes[3].bar(x, Y_norm)
axes[3].set_ylabel('Costs')
axes[3].set_title('Actual Cost Distribution of different houses')

#Predict the values of houses in $10M
Y_Pred_Final = w[0]+w[1]* (mydata.iloc[:,0]/(np.max(mydata.iloc[:,0])))+w[2]*(mydata.iloc[:,1]/(np.max(mydata.iloc[:,1])))

#Plot the predicted distribution of costs as given in the training set in $10M
axes[4].bar(x, Y_Pred_Final)
axes[4].set_ylabel('Costs')
axes[4].set_title('Actual Cost Distribution of different houses')