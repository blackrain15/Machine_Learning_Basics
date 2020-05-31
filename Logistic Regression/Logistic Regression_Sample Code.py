import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Store the training data in a Pandas dataframe. The datasheet has 3 columns - including 2 features x1 = score in Physics, x2 = score in Math and output variable y = University selection status
#Target y is a categorical variable - with 2 distinct values i.e. 1 = Selected and 0 = Not Selected

mydata = pd.read_csv("ex2data1.txt", header = None)

#Create 2 seperate dataframes for Selected and Not Selected candidates
mydata_Selected = mydata[mydata[2]==1]
mydata_NotSelected = mydata[mydata[2]==0]

#Plot the values Scores in Physics and Mathetics - Mark Selected and Not Selected candidates in 2 different markers
fig, axes = plt.subplots(nrows=2, ncols =2, figsize = (8,8))
axes[0][0].scatter(mydata_Selected.iloc[:,0], mydata_Selected.iloc[:,1], color = 'b', marker = 'x', label = 'Selected')
axes[0][0].scatter(mydata_NotSelected.iloc[:,0], mydata_NotSelected.iloc[:,1], color = 'r', marker = 's', label = 'Not Selected')

axes[0][0].set_xlabel("Score in Physics")
axes[0][0].set_ylabel("Score in Maths")
axes[0][0].set_title("Scores vs. Selection")
axes[0][0].legend(loc=0)

axes[0][0].set_xlim([min(mydata.iloc[:,0]),max(mydata.iloc[:,0])+1])
axes[0][0].set_ylim([min(mydata.iloc[:,1]),max(mydata.iloc[:,1])+1])


plt.tight_layout()

#Random initialization of x co-efficients i.e. theta0,theta1 and theta2
thetas = [0.2,0.2,-24]

#Define hyper-parameters learning rate alpha and initialize with a value. The value to be fine-tuned if used against a different training set
alpha = 0.000001

#Define a variable to store the cost function results against different iterations
cost_series = []

#Iteratively optimize the values of co-efficients thetas until the cost function converges
counter = 0
while(True):
    counter = counter+1

    cost=0    
    for i in range(len(mydata)):
        #Derive the value of z as theta0 * x0 + theta1 * x1 + theta2 * x2
        z = thetas[0]*mydata.iloc[i,0]+thetas[1]*mydata.iloc[i,1]+thetas[2]
        #Apply non-linearity using Sigmoid function to predict the value of hypothesis function
        sig_z = 1/(1+np.exp(-z))
        y_pred = sig_z
       
        #Compute cost for the current iteration
        cost = cost + (mydata.iloc[i,2]*np.log(y_pred) + (1-mydata.iloc[i,2])*np.log(1-y_pred))
    cost = cost*((-1)/len(mydata))
 
#Used for unit testing:  print("Value of Y_Pred {}, Value of Cost {} :".format(y_pred, cost))
        
    #Keeping a track of costs computed in different iterations        
    cost_series.append(cost)
    
    #Compute the value of delta for theta0 following Gradient Descent
    del_theta0 = 0
    for i in range(len(mydata)):
        z = thetas[0]*mydata.iloc[i,0]+thetas[1]*mydata.iloc[i,1]+thetas[2]
        sig_z = 1/(1+np.exp(-z))
        y_pre = sig_z
        del_theta0 = del_theta0 + (y_pred - mydata.iloc[i,2])*mydata.iloc[i,0]*(1/len(mydata))

    #Compute the value of delta for theta1 following Gradient Descent    
    del_theta1 = 0
    for i in range(len(mydata)):
        z = thetas[0]*mydata.iloc[i,0]+thetas[1]*mydata.iloc[i,1]+thetas[2]
        sig_z = 1/(1+np.exp(-z))
        y_pred = sig_z
        del_theta1 = del_theta1 + (y_pred - mydata.iloc[i,2])*mydata.iloc[i,1]*(1/len(mydata))
    
    #Compute the value of delta for theta2 following Gradient Descent
    del_theta2 = 0
    for i in range(len(mydata)):
        z = thetas[0]*mydata.iloc[i,0]+thetas[1]*mydata.iloc[i,1]+thetas[2]
        sig_z = 1/(1+np.exp(-z))
        y_pred = sig_z
        del_theta2 = del_theta2 + (y_pred - mydata.iloc[i,2])*(1/len(mydata))

    #Revisit the values of thetas - following Gradient Descent principle
    thetas[0] = thetas[0] - alpha*del_theta0
    thetas[1] = thetas[1] - alpha*del_theta1
    thetas[2] = thetas[2] - alpha*del_theta2

    #If we find that the change is cost function is <0.0000005 i.e. cost function has converged, we can stop iterating further
    #The value of 0.0000005 to be changed based of the training dataset and type of implementation
    if(counter>=2):
        if(abs(cost_series[counter-2] - cost_series[counter-1])<0.0000005):
            break
        elif(counter>500): #If cost function does not converge even after 500 iteration, quit
            break


#Plot how the costs vary with number of iterations. This is important to find out if G.D. is able to converge
axes[1][0].plot(cost_series)
axes[1][0].set_xlabel("# of iterations")
axes[1][0].set_ylabel("Cost")
axes[1][0].set_title("Cost vs. Iterations")
plt.tight_layout()


#Using the values of thetas, as derived using Gradient Descent, we can compute the values of Y_Predicted
predict_y =[]
for i in range(len(mydata)):
    z = (thetas[0]*mydata.iloc[i,0]+thetas[1]*mydata.iloc[i,1]+thetas[2])
    sig_z = 1/(1+np.exp(-z))
    if(sig_z>=0.5):     #If sigmoid of z is >=0.5, we can consider the candidate to be selected i.e. Y_Pred = 1
        predict_y.append(1)
    else:
        predict_y.append(0) #If sigmoid of z is <0.5, we can consider the candidate to be Not selected i.e. Y_Pred = 0
    
my_predicted_data = mydata.copy() #In Predicted dataframe, the values of x1 and x2 will not change
my_predicted_data.iloc[:,2] = predict_y #Replacing Y column with the predicted values

#Plot the values of Selected and Not-Selected Candidates
axes[0][1].scatter(my_predicted_data[my_predicted_data[2]==1].iloc[:,0], my_predicted_data[my_predicted_data[2]==1].iloc[:,1], color = 'b', marker = 'x', label = 'Selected')
axes[0][1].scatter(my_predicted_data[my_predicted_data[2]==0].iloc[:,0], my_predicted_data[my_predicted_data[2]==0].iloc[:,1], color = 'r', marker = 's', label = 'Not Selected')

axes[0][1].set_xlim([min(mydata.iloc[:,0]),max(mydata.iloc[:,0])+1])
axes[0][1].set_ylim([min(mydata.iloc[:,1]),max(mydata.iloc[:,1])+1])

axes[0][1].set_xlabel("Score in Physics")
axes[0][1].set_ylabel("Score in Maths")
axes[0][1].set_title("Scores vs. Predicted Selection")
axes[0][1].legend(loc=0)

#Derive the decision boundary setting z = theta0*x0+theta1*x1+theta2*x2 = 0
p1 = [0, -thetas[2]/thetas[1]]
p2 = [-thetas[2]/thetas[0], 0]
axes[0][1].plot([p1[0],p2[0]],[p1[1],p2[1]]) #Include the decision boundary in the plot