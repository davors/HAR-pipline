# Importing numpy 
import numpy as np

# Importing Pandas Library 
import pandas as pd

#    FUNCTION: import_raw_labels_file(path,columns)
#    #######################################################################
#    #      1- Import labels.txt                                           #
#    #      2- convert data from txt format to int                         #
#    #      3- convert integer data to a dataframe & insert columns names  #
#    #######################################################################  
def import_labels_file(path,columns):
    ######################################################################################
    # Inputs:                                                                            #
    #   path: A string contains the path of "labels.txt"                                 #
    #   columns: A list of strings contains the columns names in order.                  #
    # Outputs:                                                                           #
    #   dataframe: A pandas Dataframe contains labels  data in int format                #
    #             with columns names.                                                    #
    ######################################################################################
    
    
    # open the txt file
    labels_file =open(path,'r')
    
    # creating a list 
    labels_file_list=[]
    
    
    #Store each row in a list ,convert its list elements to int type
    for line in labels_file:
        labels_file_list.append([int(element) for element in line.split()])
    # convert the list of lists into 2D numpy array 
    data=np.array(labels_file_list)
    
    # Create a pandas dataframe from this 2D numpy array with column names 
    data_frame=pd.DataFrame(data=data,columns=columns)
    
    # returning the labels dataframe 
    return data_frame

#    FUNCTION: import_raw_signals(path,columns)
#    ###################################################################
#    #           1- Import acc or gyro file                            #
#    #           2- convert from txt format to float format            #
#    #           3- convert to a dataframe & insert column names       #
#    ###################################################################                      

def import_raw_signals(file_path,columns):
    ######################################################################################
    # Inputs:                                                                            #
    #   file_path: A string contains the path of the "acc" or "gyro" txt file            #
    #   columns: A list of strings contains the column names in order.                   #
    # Outputs:                                                                           #
    #   dataframe: A pandas Dataframe contains "acc" or "gyro" data in a float format    #
    #             with columns names.                                                    #
    ######################################################################################


    # open the txt file
    opened_file =open(file_path,'r')

    # Create a list
    opened_file_list=[]
    
    # loop over each line in the opened_file
    # convert each element from txt format to float 
    # store each raw in a list
    for line in opened_file:
        opened_file_list.append([float(element) for element in line.split()])

    # convert the list of lists into 2D numpy array(computationally efficient)
    data=np.array(opened_file_list)


    # Create a pandas dataframe from this 2D numpy array with column names
    data_frame=pd.DataFrame(data=data,columns=columns)

    # return the data frame
    return data_frame