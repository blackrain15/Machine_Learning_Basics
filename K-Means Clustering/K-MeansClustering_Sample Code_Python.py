import matplotlib.pyplot as plt
import numpy as np
import sys


#Let us begin with taking the number of training records from the user
print("Enter number of training records: ")
m = int(input())

if(m<30):
    print("Number of training records must be at least 30 or more!")
    sys.exit(1)
    

#Initializing the features x (# of items) and y (costs) with random values
x = np.arange(1,m+1)
y = np.zeros(m)

vals_to_gen = m
counter = 1
increment = 10
x_pos = 0
y_pos = 10
 
while(vals_to_gen>0):
   
    y[x_pos:y_pos] = np.random.randint((100*counter),(200*counter-1),increment)
    
    vals_to_gen -= 10
    
    if(vals_to_gen>=10):
        increment = 10
        x_pos += 10
        y_pos += 10
    else:
        increment = 10-vals_to_gen
        x_pos += 10
        y_pos += increment
                
    counter += 1
    
# Use Matplotlib package to draw the initial dataset
fig, axes = plt.subplots(nrows=1, ncols =1, figsize = (12,8))
axes.scatter(x, y, color = 'black')
axes.set_xlabel("Items")
axes.set_ylabel("Cost")
axes.set_title("Items vs. Cost")

#Show the graph in the console and wait for user input to proceed
plt.show()
plt.pause(0.01)
input()

# Now let us accept the number of clusters from the user
print("Enter number of clusters: ")
k = int(input())

# We shall accept only 2 or more number of clustering
if(k<2):
    print("Number of clusters must be at least 2 or more!")
    sys.exit(1)

n = 2 # n = Number of features (x,y) are 2

C_i = np.zeros(m) #Indicates which training set belongs to which cluster
mu_k = np.zeros((k,n)) #Indicates the x,y coordinates of the cluster centroids
mu_c_i = np.zeros((m,n)) #Indicates the cluster centroids of each of the training set


#Random selection of the existing training datapoints as given number of cluster centroids
for i in range(k):
    while(True):  #Avoids selecting the same x,y datapoints as centroids of 2 different clusters
        index = np.random.randint(0,m-1)
        if([x[index],y[index]] not in mu_k):
            mu_k[i,:] = [x[index],y[index]]
            break


#Use Matplotlib package to draw the datapoints with the clustet centroids (initial selection) marked in Red
fig, axes = plt.subplots(nrows=1, ncols =1, figsize = (12,8))
axes.scatter(x, y, color = 'black')
axes.scatter(mu_k[:,0],mu_k[:,1] , color = 'r')
axes.set_xlabel("Items")
axes.set_ylabel("Cost")
axes.set_title("Items vs. Cost")

#Show the plot and wait for user input to proceed further
plt.show()
plt.pause(0.01)
input()

#Randomly assign RGB values to each of the clusters
my_colors = []
for i in range(k):
    my_colors.append(np.array([np.round(np.random.random(),1),np.round(np.random.random(),1),np.round(np.random.random(),1)]))


#Let us incremental iterate and keep on shifting the cluster centroids until the centroids stop shifting

Flag = True
while(Flag):
    dist_to_centroids = np.zeros(k)

#Iteratively decide the distances of each of the data points from cluster centroids
    for i in range(m):
        j = 0
        min_dist = np.power((mu_k[j,0]-x[i]),2)+np.power((mu_k[j,1]-y[i]),2)
        mu_c_i[i,0] = mu_k[j,0]
        mu_c_i[i,1] = mu_k[j,1]
        j +=1
        C_i[i] = j

#Assign each of the data points to the nearest clusters (with respect to the corresponding cluster centroid)
        while(j<k):
            dist_to_centroids[j] = np.power((mu_k[j,0]-x[i]),2)+np.power((mu_k[j,1]-y[i]),2)
            
            if(dist_to_centroids[j]<min_dist):
                min_dist = dist_to_centroids[j]
                C_i[i] = j+1
                mu_c_i[i,0] = mu_k[j,0]
                mu_c_i[i,1] = mu_k[j,1]
            
            j+=1

#Plot the training set with the clusters in varying colors
    fig, axes = plt.subplots(nrows=1, ncols =1, figsize = (12,8))
    for i in range(k):
        for j in range(m):
            if(C_i[j]==i+1):
                axes.scatter(x[j], y[j], color = my_colors[i].reshape(1,3))
                
#Plot of cluster centroids as big colorful spots
    for i in range(k):    
        axes.scatter(mu_k[i,0],mu_k[i,1] , color = my_colors[i].reshape(1,3), lw=7)
    
    axes.set_xlabel("Items")
    axes.set_ylabel("Cost")
    axes.set_title("Items vs. Cost")

#Show the plot and wait for user input to proceed further
    plt.show()
    plt.pause(0.01)


#Update the list of cluster centroids as the mean values of the cluster specific datapoints    
    for i in range(k):
        sum_x = 0
        sum_y = 0
        counter = 0
        for j in range(m):
            if(C_i[j]==i+1):
                counter += 1
                sum_x += x[j]
                sum_y += y[j]

#If the cluster centroids do not change, stop iterating, otherwise, proceed with new iteration
        if(mu_k[i,0] == int(sum_x/counter)):
            Flag = False
        else:
            Flag = True
            mu_k[i,0] = int(sum_x/counter)
    
        if(mu_k[i,1] == int(sum_y/counter)):
            Flag = False
        else:
            Flag = True
            mu_k[i,1] = int(sum_y/counter)

#Wait for user input before proceeding with next iteration
    input()