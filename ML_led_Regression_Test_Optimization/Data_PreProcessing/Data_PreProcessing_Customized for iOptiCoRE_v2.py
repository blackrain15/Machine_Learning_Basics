import argparse
import numpy as np
import pandas as pd
import os
import sys

def process_input_files(input_path = os.getcwd()+"/" ):

    #Storing the 4 input training datasheets in 4 Pandas dataframes
    module_list = pd.read_csv(input_path+"Module_List.csv")
    module_mapping = pd.read_csv(input_path+"Module_Mapping.csv")
    test_cases = pd.read_csv(input_path+"Test_Cases.csv")
    release_data = pd.read_csv(input_path+"Release_Data.csv")
    
    #Storing the input release information in a different dataframe
    input_release = pd.read_csv(input_path+"Input_Release.csv")
    
    #Replacing the missing or null values with blanks
    module_list = module_list.fillna(value = "")
    module_mapping = module_mapping.fillna(value = "") 
    test_cases = test_cases.fillna(value = "")
    release_data = release_data.fillna(value = "")
    
    #Error handling for Module list
    module_list['Module Name'] = list(map(lambda x: x.upper().replace(" ",""), module_list['Module Name']))  #Remove whitespaces and converts all chars to uppercase
    
    module_list.sort_values('Module Name',inplace = True, ascending=True) #Sort the list in ascending order
    
    if(module_list['Module Name'].nunique() != len(module_list)):  #Check if there is any duplicate module name
        print("There are duplicate mododule names in Module_List.csv!")
        sys.exit(1)
    
    #Error handling for Module Mapping
    module_mapping['Mod_Name'] = list(map(lambda x: x.upper().replace(" ",""), module_mapping['Mod_Name'])) #Remove whitespaces and coverts all chars to uppercase for all rows
    module_mapping.set_index("Mod_Name", inplace = True)
    
    module_mapping.columns = map(lambda x: x.upper().replace(" ",""),module_mapping.columns) #Remove whitespaces and converts all chars to uppercase for all columns
    
    module_mapping.sort_index(axis = 0, inplace = True, ascending = True) #Sorts the grid row-wise
    module_mapping.sort_index(axis = 1, inplace = True, ascending = True) #Sorts the grid column-wise
    
    col_names = list(module_mapping.columns.values) #Module names across columns
    index_names = list(module_mapping.index.values) #Module names across rows
    
    module_names = col_names
    
    if((col_names != index_names) or (col_names != list(module_list['Module Name']))):   #Row and column names must match and also the entries should be part of module list
        print("The row and column values do not match with the ones provided in module list")
        sys.exit(1)
    
    #Error handling for TC sheet
    for i in range(len(test_cases.columns.values)):
        test_cases.iloc[:,i] = list(map(lambda x: x.upper().replace(" ",""), test_cases.iloc[:,i])) #Remove whitespaces and converts all chars to uppercase
    
    for i in range(len(test_cases)):  #Create a list of linked modules
        test_cases.iloc[i,5] = test_cases.iloc[i,5].split('|')
    
    TC_IDs = list(test_cases.iloc[:,0])  #Master list of Test cases. Will be referred in multiple places henceforth
    
    for t in test_cases['Module Names']: #Error out any TC having no module
        if(len(t)==0):
            print("One or more test cases are not mapped with any module!")
            sys.exit(1)
    
    if(len(test_cases[test_cases['Manual Test Case ID']==""])>0):  #Error out any TC having no ID
        print("One or more test cases have no test case ID!")
        sys.exit(1)    
    
    counter = len(test_cases[test_cases['Complexity']=="HIGH"])+len(test_cases[test_cases['Complexity']=="MEDIUM"])+len(test_cases[test_cases['Complexity']=="LOW"]) #Count of rows having legitimate Complexity markings
    
    if(counter!=len(test_cases)):  #Error if one of more rows have invalid complexity value
        print("Invalid values against Complexity column!")
        sys.exit(1)
    
    counter = len(test_cases[test_cases['Business Priority']=="HIGH"])+len(test_cases[test_cases['Business Priority']=="MEDIUM"])+len(test_cases[test_cases['Business Priority']=="LOW"]) #Count of rows having legitimate Business Priority markings
    
    if(counter!=len(test_cases)): #Error if one of more rows have invalid priority value
        print("Invalid values against Business Priority column!")
        sys.exit(1)
    
    flag = False
    for i in range(len(test_cases)):
        mod_list = test_cases.iloc[i,5]
        for mod in mod_list:
            if(mod not in module_names):  #Check if all the module names mapped with TCs are part of module list
                flag = True
    
    if(flag):  #Error if the modules mapped with TCs are not part of module list
        print("Incorrect module name against one or more test cases")
        sys.exit(1)
        
    #Error handling for Release sheet
    for i in range(len(release_data.columns.values)):
        release_data.iloc[:,i] = list(map(lambda x: x.upper().replace(" ",""), release_data.iloc[:,i])) #Removes whitespaces and converts all chars to uppercase
    
    for i in range(len(release_data)): #Create a list of linked modules
        release_data.iloc[i,3] = release_data.iloc[i,3].split('|')
    
    for i in range(len(release_data)): #Create a list of linked TCs
        release_data.iloc[i,4] = release_data.iloc[i,4].split('|')
    
    
    if(len(release_data[release_data['Release Name']==""])>0):  #Error out any release having no name is present
        print("One or more releases have no Release Name!")
        sys.exit(1)
    
    for r in release_data['Impacted Modules'] :  #Error out any release having no name is present
        if(len(r)==0):
            print("One or more releases have no module mapping!")
            sys.exit(1)
    
    flag = False
    for i in range(len(release_data)):
        mod_list = release_data.iloc[i,3]
        for mod in mod_list:
            if(mod not in module_names): #Check if all the module names mapped with releases are part of module list
                flag = True
    
    if(flag): #Error if the modules mapped with releases are not part of module list
        print("One or more impacted modules are not part of module list!")
        sys.exit(1)
    
    flag = False
    for i in range(len(release_data)):
        TC_ID_list = release_data.iloc[i,4]
        for TC in TC_ID_list:
            if(TC != "" and TC not in TC_IDs):  #Check if all the TC IDs mapped with releases are part of TC list
                flag = True
    
    if(flag): #Error if the TCs mapped with release are not part of TC list
        print("One or more failed test cases are not part of Test Cases list!")
        sys.exit(1)
    
    unique_dev_types = set(release_data.iloc[:,2]) #Create a list of unique release names
    unique_dev_types = list(unique_dev_types)
    
    #Error handling for Input Release sheet
    if(len(input_release)>1):
        print("The input sheet should not have more than one release information!")
        sys.exit(1)
    
    for i in range(len(input_release.columns.values)):
        input_release.iloc[:,i] = list(map(lambda x: x.upper().replace(" ",""), input_release.iloc[:,i])) #Removes whitespaces and converts all chars to uppercase
    
    input_release.iloc[0,3] = input_release.iloc[0,3].split('|')
    
    if(len(input_release[input_release['Release Name']==""])>0):  #Error out any release having no name is present
        print("The input release name is missing!")
        sys.exit(1)
    
    r = input_release['Impacted Modules']  #Error out the input release does not have impacted modules attached to it
    
    if(len(r)==0):
        print("The input release does not have module mapping!")
        sys.exit(1)
    
    flag = False
    mod_list = input_release.iloc[0,3]
    for mod in mod_list:
        if(mod not in module_names): #Check if all the module names mapped with the release are part of module list
            flag = True
    
    if(flag): #Error if the modules mapped with the release are not part of module list
        print("One or more impacted modules in the input release sheet are not part of module list!")
        sys.exit(1)
    
    unique_dev_types.append(input_release.iloc[0,2])
    unique_dev_types = list(set(unique_dev_types))
    
    #Creating column names for the Output dataframe
    TC_Mod_impact_frame = []
    
    TC_Mod_impact_frame.append("Release") #Column 1 is for release names
    TC_Mod_impact_frame.append("Development") #Column 2 is for release types
    TC_Mod_impact_frame.append("TechAndDomainComplexity") #Column 3 is for TechDomainComplexity - No longer used, default to HIGH
    
    for i in range(len(module_names)): #Followed by module names
        TC_Mod_impact_frame.append(module_names[i])
    
    for i in range(len(TC_IDs)):
        TC_Mod_impact_frame.append(TC_IDs[i]+"_M") #Followed by Test Case ID_Module mapping column
        TC_Mod_impact_frame.append(TC_IDs[i]+"_S") #Followed by Test Case ID_Status column
    
    
    data_filler = np.zeros(len(TC_Mod_impact_frame)*(len(release_data)+1)) #Populate the entire output datasheet with zeros. One additional row created for the New Release entry (to be predicted by ML)
    data_filler = data_filler.reshape((len(release_data)+1),len(TC_Mod_impact_frame)) #Change the matrix in line with the output dimensions
    
    output_sheet = pd.DataFrame(columns = TC_Mod_impact_frame, data = data_filler) #Crate a dataframe for easy handling of data and eventually, dumping into an output CSV
    
    for i in range(len(release_data)): #Populate the rows in line with the training data (i.e. release )

        output_sheet.iloc[i,0] = release_data.iloc[i,0] #Populate release name in column 0
        output_sheet.iloc[i,1] = release_data.iloc[i,2] #Populate release type in column 1
        output_sheet.iloc[i,2] = "HIGH" #Populate TechDomainComplexity in column 3 - default value = HIGH
        j=3 #Since first 3 columns are populated now, we can start from column index 3
        counter = 0
        while(j < (len(module_names)+3)): #Depending upon the set of modules impacted in each of the release data, appropriate module column to be marked as 1. Rest all are 0.
            mod_list_direct = release_data.iloc[i,3]
            mod_list_overall = []
            for mod_l in mod_list_direct:
                mod_list_overall += list(module_mapping[module_mapping[mod_l]==1].index.values)
            if(module_names[counter] in mod_list_overall):
                output_sheet.iloc[i,j] = 1
            else:
                output_sheet.iloc[i,j] = 0
            j += 1
            counter += 1
        
        counter_TC_M = 3 + len(module_names) #Starting index of test case_moddule map columns. First 3 columns are Release Name, Release Type and TechDomainComplexity
        counter_TC_S = counter_TC_M+1 #Starting index of test case_status columns
        
        while(counter_TC_S < len(TC_Mod_impact_frame)): #Let us start by populating the test case_status columns
            tc_list_temp = release_data.iloc[i,4]
            if(TC_Mod_impact_frame[counter_TC_S][0:-2] in tc_list_temp): #If the test case ID is amongst the list of failed TCs for the given release,populate 1, else, populate 0
                output_sheet.iloc[i,counter_TC_S] = 1
            else:
                output_sheet.iloc[i,counter_TC_S] = 0
            counter_TC_S += 2 #Increment=2 as we are skipping the test case_module mapping columns
        
        while(counter_TC_M < (len(TC_Mod_impact_frame)-1)):
            mod_list_temp = release_data.iloc[i,3]
            
            tc_mod_direct_map = list(test_cases[test_cases['Manual Test Case ID']==TC_Mod_impact_frame[counter_TC_M][0:-2]].iloc[0,5])
            tc_mod_overall_map = []
            for m_d in tc_mod_direct_map:
                tc_mod_overall_map += list(module_mapping[module_mapping[m_d]==1].index.values)
            
            Flag = False
            for m_o in tc_mod_overall_map: #If the modules linked with the test case IDs (directly or indirectly via module mapping) is amongst the list of impacted modules for the given release,Flag = True, else, Flag = False
                if(m_o in mod_list_temp):
                    Flag = True
                    break
            
            if(Flag): #If the release impacted module(s) is linked with the given test case, populate 1, else, populate 0.
                output_sheet.iloc[i,counter_TC_M] = 1
            else:
                output_sheet.iloc[i,counter_TC_M] = 0
            
            counter_TC_M += 2 #Increment=2 as we are skipping the test case_status columns
    
    #Now fill up the last row of the output sheet - which is for the input release
            
    i = len(release_data)
    
    output_sheet.iloc[i,0] = input_release.iloc[0,0]  #Populate release name in column 0
    output_sheet.iloc[i,1] = input_release.iloc[0,2] #Populate release type in column 1
    output_sheet.iloc[i,2] = "HIGH" #Populate TechDomainComplexity in column 3 - default value = HIGH
    
    j=3 #First 3 columns are for Release, Development Type & TechDomainComplexity
    counter = 0
    
    while(j < (len(module_names)+3)): #Depending upon the set of modules impacted in each of the release data, appropriate module column to be marked as 1. Rest all are 0.
        mod_list_direct = input_release.iloc[0,3]
        mod_list_overall = []
        for mod_l in mod_list_direct:
            mod_list_overall += list(module_mapping[module_mapping[mod_l]==1].index.values)
        if(module_names[counter] in mod_list_overall):
            output_sheet.iloc[i,j] = 1
        else:
            output_sheet.iloc[i,j] = 0
        j += 1
        counter += 1
    
    counter_TC_M = 3 + len(module_names) #Starting index of test case_moddule map columns, First 3 columns are for Release, Development Type & TechDomainComplexity

    output_sheet[TC_Mod_impact_frame[3::]] = output_sheet[TC_Mod_impact_frame[3::]].astype(int) #Use a slicer to avoid getting the 0,1 values in float format
    
    while(counter_TC_M < (len(TC_Mod_impact_frame)-1)):
        mod_list_temp = input_release.iloc[0,3]
            
        tc_mod_direct_map = list(test_cases[test_cases['Manual Test Case ID']==TC_Mod_impact_frame[counter_TC_M][0:-2]].iloc[0,5])
        tc_mod_overall_map = []
        for m_d in tc_mod_direct_map:
            tc_mod_overall_map += list(module_mapping[module_mapping[m_d]==1].index.values)
            
        Flag = False
        for m_o in tc_mod_overall_map: #If the modules linked with the test case IDs (directly or indirectly via module mapping) is amongst the list of impacted modules for the given release,Flag = True, else, Flag = False
            if(m_o in mod_list_temp):
                Flag = True
                break
            
        if(Flag): #If the release impacted module(s) is linked with the given test case, populate 1, else, populate 0.
            output_sheet.iloc[i,counter_TC_M] = 1
        else:
            output_sheet.iloc[i,counter_TC_M] = 0
        
        output_sheet.iloc[i,counter_TC_M+1] = ""
        counter_TC_M += 2 #Increment=2 as we are skipping the test case_status columns
    
    
    output_sheet.to_csv(input_path+"Output_sheet.csv", index=False, sep = "|", quoting = 1) #Generate output with pipeline delimted and text qualifier as "


######################################################
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--File_path", help = "Enter the folder path where the training data and input file is present")
    args = parser.parse_args()
    if(len(sys.argv)>1):
        input_path = args.File_path.replace("\\","/")
        input_path += "/"
        process_input_files(input_path)
    else:
        process_input_files()

