import argparse
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import sys
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def process_output_file(input_path = os.getcwd()+"/" ):
 
    #Open the output_sheet generated post data cleansing process
    output_sheet = pd.read_csv(input_path+"Output_sheet.csv")
    
    col_names = list(output_sheet.columns.values) #Columns names in the output sheet
    
    #Identify till which column the release type and module impact specific input data is present i.e. the column after which the TC specific pair of columns with suffix _M and _S starts
    flag = True
    counter = 0
    while(flag):
        if(col_names[counter][-2:]=="_M"):
            flag = False
        else:
            counter +=1
    
    #Identify how many records exist in the output sheet
    rec_count = len(output_sheet)
    
    #Create a list of input features - covering all the columns of release type and release level module impact specific inputs
    input_cols = col_names[0:counter]
    input_counter = counter
    
    #Add up the test case specific module impact columns which are input features
    while(input_counter<len(col_names)):
        input_cols.append(col_names[input_counter])
        input_counter+=2
    
    #Create a list of output values - which are columns with the test case status values i.e. columns having _S suffix
    output_cols = []
    output_counter = counter+1
    while(output_counter<len(col_names)):
        output_cols.append(col_names[output_counter])
        output_counter+=2
        
    #Using the relevant column names as stored in input_cols and output_cols lists, slice the pandas dataframe to store input features as well as output test case status    
    X_inputTrainFeatures = np.array(output_sheet[input_cols].iloc[0:rec_count-1,:]) #The last row is the one to use for Prediction - hence, taking rows till rec_count - 1
    y_OutputTrainStatus  = np.array(output_sheet[output_cols].iloc[0:rec_count-1,:])
    
    #No. of input nodes in the neural network = No. of columns treated as input features (i.e. release type, module impact and TC specific module linkage columns)
    _,input_nodes = X_inputTrainFeatures.shape
    #No. of output nodes in the neural network = No. of columns treated as output variables (i.e, TC specific status columns)
    _,output_nodes = y_OutputTrainStatus.shape
    
    
    #Create a numpy array of a single row with the columns set as the test case specific status (_S) columns
    X_inputToPredFeatures = np.array(output_sheet[input_cols].iloc[rec_count-1,:]).reshape(1,input_nodes)
    
    #Creation of a fully connected neural network model
    model = tf.keras.Sequential()
    #Define input layer with no. of nodes = input_nodes and first hidden layer with no. of nodes = 2 x input_nodes. Activation function as Sigmoid.
    model.add(tf.keras.layers.Dense(2*input_nodes, input_dim=input_nodes, activation = 'sigmoid'))
    #Define second hidden layer with no. of nodes =  2 x input_nodes. Activation function as Sigmoid.
    model.add(tf.keras.layers.Dense(2*input_nodes, activation = 'sigmoid'))
    #Define output layer with activation function as Sigmoid
    model.add(tf.keras.layers.Dense(output_nodes, activation = 'sigmoid'))
    
    #Compile the model with Stochastic Gradient Boosted optimizer, loss/cost function as binary cross entropy and evaluation metric as Recall i.e. Total Predicted Failure TCs which actuall failed/(Total Predicted Failure TCs which actually failed + Total Predicted Passed TCs which actually Failed)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    #We shall follow Train-Hold out split to perform validation of the model. Let us ensure that our validation split has at least 2 records
    if(rec_count*0.2 <2):
        test_sample_size = 2/rec_count
    else:
        test_sample_size = 0.2
    
    #Perform model fitment with the training data. Batch size is 4 datapoints for batch gradient descent based optimization - along with total iteration of 1000.
    history = model.fit(X_inputTrainFeatures, y_OutputTrainStatus, epochs=1500, batch_size=4, validation_split=test_sample_size) #history object stores loss, acc, val_loss, val_acc against each epoch (iteration)
    
    #Debugging step: _, accuracy = model.evaluate(X_inputTrainFeatures, y_OutputTrainStatus)
    
    #Enter the current release datapoints in the model to predict TC specific failue probabilities
    y_OutputToPredStatus_ANN = model.predict(X_inputToPredFeatures).reshape(output_nodes,1)
    
    
    #Storing average accuracy as observed in the validation samples across the epochs/iterations     
    ANN_accuracy = np.mean(history.history['val_acc'])
    
    #Create a data model using Suppoer Vector Regression
    SVR_model = SVR(kernel='rbf') #Going with the default values of C and gamma
    multioutput_wrapper = MultiOutputRegressor(SVR_model) #Need to form a MultiOutputRegressor as the output y has multiple features (i.e. status of multiple test cases)
    multioutput_wrapper.fit(X_inputTrainFeatures, y_OutputTrainStatus) #Training the model with the dataset
    y_OutputToPredStatus_SVM = multioutput_wrapper.predict(X_inputToPredFeatures) #The last record (of current release) is used for prediction
    
    #Split the overall sample size into train vs. hold-out/test
    X_train, X_test, y_train, y_test = train_test_split(X_inputTrainFeatures, y_OutputTrainStatus, test_size=test_sample_size, shuffle=True)
    multioutput_wrapper.fit(X_train, y_train) #Model training using training dataset
    
    #For the test samples, calculate the accuracy scores at test case level. Overall accuracy found with SVM will be the average accuracy score found across test cases in the test samples    
    SVM_accuracy = 0
    for i in range(len(X_test)):
        yhat_test = multioutput_wrapper.predict(X_test[i].reshape(1,input_nodes))
        yhat_test = np.ceil(yhat_test.reshape(output_nodes))
    SVM_accuracy += accuracy_score(yhat_test, y_test[i])
    
    SVM_accuracy = SVM_accuracy/i
    
    #We shall go with the predicted values from the model with higher accuracy
    if(SVM_accuracy > ANN_accuracy):
        y_OutputToPredStatus = y_OutputToPredStatus_SVM
    else:
        y_OutputToPredStatus = y_OutputToPredStatus_ANN
    
    #Machine learning led operations end here. Now we shall process the output datasheet in proper format
    
    #Create a Numpy array to store test case IDs
    y_OutputToPredTCNames = []
    for i in range(output_nodes):
        y_OutputToPredTCNames.append(str(output_cols[i][0:-2]))
    y_OutputToPredTCNames = np.array(y_OutputToPredTCNames).reshape(output_nodes,1)
    
    #Open Test Cases excel and store in a Pandas dataframe
    test_cases = pd.read_csv(input_path+"Test_Cases.csv")
    test_cases = test_cases.fillna(value = "")
    test_cases = test_cases.astype('str')
    
    #Error handling for TC sheet
    for i in range(len(test_cases.columns.values)):
        test_cases.iloc[:,i] = list(map(lambda x: x.upper().replace(" ",""), test_cases.iloc[:,i])) #Remove whitespaces and converts all chars to uppercase
    
    #Open Release excel and store in a Pandas dataframe
    release_data = pd.read_csv(input_path+"Release_Data.csv")
    release_data = release_data.fillna(value = "")
    
    #Error handling for Release sheet
    for i in range(len(release_data.columns.values)):
        release_data.iloc[:,i] = list(map(lambda x: x.upper().replace(" ",""), release_data.iloc[:,i])) #Removes whitespaces and converts all chars to uppercase
    
    #Pipeline delimited test case IDs in the 'Manual Failed Test Case ID' column to be made lists
    for i in range(len(release_data)): #Create a list of linked TCs
        release_data.iloc[i,4] = release_data.iloc[i,4].split('|')
    
    #Create a list of test cases with failure history
    TC_with_failhistory = []
    for i in range(len(release_data)):
        temp = list(release_data.iloc[i,4])
        for tc in temp:
            if(tc != ''):
                TC_with_failhistory.append(tc)
    
    #Create a disctory with the unique test cases along with failure frequency
    TC_withFailhistfreq = {} 
    for item in TC_with_failhistory: 
        if (item in TC_withFailhistfreq): 
            TC_withFailhistfreq[item] += 1
        else: 
            TC_withFailhistfreq[item] = 1
    
    #Crete a 2D Numpy array to store all TC IDs in column 1, Count of TC failures in column 2
    y_Output_TC_FailFreq = np.empty((output_nodes,2),dtype = 'object')
    y_Output_TC_FailFreq[:,0] = y_OutputToPredTCNames[:,0]
    
    for i in range(output_nodes):
        if(y_Output_TC_FailFreq[i,0] in TC_withFailhistfreq):
            y_Output_TC_FailFreq[i,1] = TC_withFailhistfreq[y_Output_TC_FailFreq[i,0]]
        else:
            y_Output_TC_FailFreq[i,1] = 0
    
    #Create numpy array to store Automation Script ID, Test Case Status, Priority, Complexity
    other_TC_attributes = []
    for i in range(len(y_OutputToPredTCNames)):
        temp = str(y_OutputToPredTCNames[i,0])
        other_TC_attributes.append(list(test_cases[test_cases['Manual Test Case ID']==temp].iloc[0,1:5]))
    
    other_TC_attributes = np.array(other_TC_attributes)
    
    
    #Create a Pandas DataFrame to append the TC Name, Auto. Script ID etc. as different columns
    prediction_sheet = pd.DataFrame(data = {'TC_Name' : y_OutputToPredTCNames[:,0], 
                                           'Auto_TC_ID': other_TC_attributes[:,0],
                                           'Status': other_TC_attributes[:,1],
                                           'Priority' : other_TC_attributes[:,2],
                                           'Severity' : other_TC_attributes[:,3],
                                           'Failure_Frequency' : y_Output_TC_FailFreq[:,1],
                                           'TC_Failure_Probability' : y_OutputToPredStatus[:,0]}, 
                                    columns = ['TC_Name','Auto_TC_ID','Status','Priority','Severity','Failure_Frequency','TC_Failure_Probability'])
    
    
    #Sort the dataframe based on descending values of TC failure probability
    prediction_sheet.sort_values('TC_Failure_Probability', inplace= True, ascending = False)
    
    #Write the pandas dataframe to an output CSV
    prediction_sheet.to_csv(input_path+"Prediction_sheet.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--File_path", help = "Enter the folder path where the training data and input file is present")
    args = parser.parse_args()
    if(len(sys.argv)>1):
        input_path = args.File_path.replace("\\","/")
        input_path += "/"
        process_output_file(input_path)
    else:
        process_output_file()