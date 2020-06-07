import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read the input file having 3 columns - Area, # of Rooms and Cost in $1000
mydata = pd.read_csv("housepricedata.csv")

layer1_vals = np.array(mydata.iloc[:,0:3])
layer1_vals = layer1_vals.astype(float)

layer1_vals[:,0] = list(map((lambda x : ((x - np.mean(layer1_vals[:,0]))/max(layer1_vals[:,0]))),layer1_vals[:,0]))
layer1_vals[:,1] = list(map((lambda x : ((x - np.mean(layer1_vals[:,1]))/max(layer1_vals[:,1]))),layer1_vals[:,1]))
layer1_vals[:,2] = list(map((lambda x : ((x - np.mean(layer1_vals[:,2]))/max(layer1_vals[:,2]))),layer1_vals[:,2]))

output_act_vals = np.array(mydata.iloc[:,3])
output_act_vals = output_act_vals.reshape(len(mydata),1)

layer2_vals = np.ones((len(mydata),5))
layer3_vals = np.ones((len(mydata),5))

output_pred_vals = np.ones((len(mydata),1))

bias_layer1_wts = np.random.random(5)
bias_layer1_wts = bias_layer1_wts.reshape(1,5)

weights_layer1 = np.random.random(15)
weights_layer1 = weights_layer1.reshape(3,5)

bias_layer2_wts = np.random.random(5)
bias_layer2_wts = bias_layer2_wts.reshape(1,5)

weights_layer2 = np.random.random(25)
weights_layer2 = weights_layer2.reshape(5,5)

bias_layer3_wts = np.random.random(1)
bias_layer3_wts = bias_layer3_wts.reshape(1,1)

weights_layer3 = np.random.random(5)
weights_layer3 = weights_layer3.reshape(5,1)

alpha = 0.000001
cost_list = []
counter=0

sigmoid_f = lambda x : (1/(1+np.exp(-x)))

while(True):

    layer2_vals = np.matmul(layer1_vals, weights_layer1)
    layer2_vals += np.matmul(np.ones((len(mydata),1)),bias_layer1_wts)
    
    for i in range(len(mydata)):
        layer2_vals[i,:] = list(map(sigmoid_f, layer2_vals[i,:]))
    
    layer3_vals = np.matmul(layer2_vals, weights_layer2)
    layer3_vals += np.matmul(np.ones((len(mydata),1)),bias_layer2_wts)

    for i in range(len(mydata)):
        layer3_vals[i,:] = list(map(sigmoid_f, layer3_vals[i,:]))
    
    output_pred_vals = np.matmul(layer3_vals, weights_layer3)
    output_pred_vals += np.matmul(np.ones((len(mydata),1)),bias_layer3_wts)

    for i in range(len(mydata)):
        output_pred_vals[i,:] = list(map(sigmoid_f, output_pred_vals[i,:]))
    
    cost = 0
    for i in range(len(mydata)):
        cost = cost + (output_act_vals[i,0]*np.log(output_pred_vals[i,0]) + (1-output_act_vals[i,0])*(1-np.log(output_pred_vals[i,0])))
    
    cost = -cost/len(mydata)
    cost_list.append(cost)
    counter+= 1
    
    del_layer4 = (output_pred_vals - output_act_vals)
    del_weights_layer3 = np.matmul(np.transpose(layer3_vals), del_layer4)
    bias_layer3 = np.ones((len(mydata),1))
    bias_layer3_wts -= alpha * np.matmul(np.transpose(bias_layer3), del_layer4)
    
    del_layer3 = np.multiply(np.matmul(del_layer4, np.transpose(weights_layer3)), np.multiply(layer3_vals, (np.ones((1460,5))-layer3_vals)))
    del_weights_layer2 = np.matmul(np.transpose(layer2_vals), del_layer3)
    bias_layer2 = np.ones((len(mydata),1))
    bias_layer2_wts -= alpha * np.matmul(np.transpose(bias_layer2), del_layer3)
    
        
    del_layer2 = np.multiply(np.matmul(del_layer3, np.transpose(weights_layer2)), np.multiply(layer2_vals, (np.ones((1460,5))-layer2_vals)))
    del_weights_layer1 = np.matmul(np.transpose(layer1_vals), del_layer2)
    bias_layer1 = np.ones((len(mydata),1))
    bias_layer1_wts -= alpha * np.matmul(np.transpose(bias_layer1), del_layer2)
    
    weights_layer1 = weights_layer1 - alpha * del_weights_layer1
    weights_layer2 = weights_layer2 - alpha * del_weights_layer2
    weights_layer3 = weights_layer3 - alpha * del_weights_layer3
   
    if(counter>2):
        if(abs(cost_list[counter-2]-cost_list[counter-1])<0.0000001):
            break
        #if(counter>200):
            #break
        
fig, axes = plt.subplots(nrows=2, ncols =1, figsize = (12,8))

axes[0].plot(cost_list, color = 'g')
axes[0].set_xlabel("# of Iters")
axes[0].set_ylabel("Cost Func.")
axes[0].set_title("Cost vs. # of Iters.")