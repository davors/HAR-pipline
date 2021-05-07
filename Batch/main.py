import numpy as np
import scipy as sp
import pandas as pd
from glob import glob

from scipy import fftpack # import fftpack to use all fft functions
from numpy.fft import *

import dataLoading as dl
import featureGeneration as fg
import filterRawData as frd

############################## Constants #############################
sampling_freq=50 # 50 Hz(hertz) is sampling frequency: the number of captured values of each axial signal per second.
freq1 = 0.3 # freq1=0.3 hertz [Hz] the cuttoff frequency between the DC compoenents [0,0.3] and the body components[0.3,20]hz
freq2 = 20  # freq2= 20 Hz the cuttoff frequcency between the body components[0.3,20] hz and the high frequency noise components [20,25] hz


def main():
    ####################### Scraping RawData files paths########################
    Raw_data_paths = sorted(glob("Data/Original-Data/HAPT-Dataset/Raw-Data/*"))
    # Selecting acc file paths only
    Raw_acc_paths=Raw_data_paths[0:61]

    # Selecting gyro file paths only
    Raw_gyro_paths=Raw_data_paths[61:122]

    # printing info related to acc and gyro files
    print (("RawData folder contains in total {:d} file ").format(len(Raw_data_paths)))
    print (("The first {:d} are Acceleration files:").format(len(Raw_acc_paths)))
    print (("The second {:d} are Gyroscope files:").format(len(Raw_gyro_paths)))
    print ("The last file is a labels file")

    # printing 'labels.txt' path
    print ("labels file path is:",Raw_data_paths[122])

    ########################################### RAWDATA DICTIONARY ##############################################################

    # creating an empty dictionary where all dataframes will be stored
    raw_dic={}


    # creating list contains columns names of an acc file
    raw_acc_columns=['acc_X','acc_Y','acc_Z']

    # creating list contains gyro files columns names
    raw_gyro_columns=['gyro_X','gyro_Y','gyro_Z']

    # loop for to convert  each "acc file" into data frame of floats and store it in a dictionnary.
    for path_index in range(0,len(Raw_acc_paths)):
            
            # extracting the file name only and use it as key:[expXX_userXX] without "acc" or "gyro"
            key= Raw_data_paths[path_index][-16:-4]
            
            # Applying the function defined above to one acc_file and store the output in a DataFrame
            raw_acc_data_frame=dl.import_raw_signals(Raw_data_paths[path_index],raw_acc_columns)
            
            # By shifting the path_index by 61 we find the index of the gyro file related to same experiment_ID
            # Applying the function defined above to one gyro_file and store the output in a DataFrame
            raw_gyro_data_frame=dl.import_raw_signals(Raw_data_paths[path_index+len(Raw_acc_paths)],raw_gyro_columns)
            
            # concatenate acc_df and gyro_df in one DataFrame
            raw_signals_data_frame=pd.concat([raw_acc_data_frame, raw_gyro_data_frame], axis=1)
            
            # Store this new DataFrame in a raw_dic , with the key extracted above
            raw_dic[key]=raw_signals_data_frame

    # raw_dic is a dictionary contains 61 combined DF (acc_df and gyro_df)
    print('raw_dic contains %d DataFrame' % len(raw_dic))

    #################################
    # creating a list contains columns names of "labels.txt" in order
    raw_labels_columns=['experiment_number_ID','user_number_ID','activity_number_ID','Label_start_point','Label_end_point']

    # The path of "labels.txt" is last element in the list called "Raw_data_paths"
    labels_path=Raw_data_paths[-1]

    # apply the function defined above to labels.txt 
    # store the output  in a dataframe 
    Labels_Data_Frame=dl.import_labels_file(labels_path,raw_labels_columns)

    # Creating a dictionary for all types of activities
    # The first 6 activities are called Basic Activities as(BAs) 3 dynamic and 3 static
    # The last 6 activities are called Postural Transitions Activities as (PTAs)
    Acitivity_labels=AL={
            1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS', # 3 dynamic activities
            4: 'SITTING', 5: 'STANDING', 6: 'LAYING', # 3 static activities
            
            7: 'STAND_TO_SIT',  8: 'SIT_TO_STAND',  9: 'SIT_TO_LIE', 10: 'LIE_TO_SIT', 
        11: 'STAND_TO_LIE', 12: 'LIE_TO_STAND',# 6 postural Transitions
        }

    time_sig_dic={} # An empty dictionary will contains dataframes of all time domain signals
    raw_dic_keys=sorted(raw_dic.keys()) # sorting dataframes' keys

    for key in raw_dic_keys: # iterate over each key in raw_dic
        
        raw_df=raw_dic[key] # copie the raw dataframe associated to 'expXX_userYY' from raw_dic
        
        time_sig_df=pd.DataFrame() # a dataframe will contain time domain signals
        
        for column in raw_df.columns: # iterate over each column in raw_df
            
            t_signal=np.array(raw_df[column]) # copie the signal values in 1D numpy array
            
            med_filtred=frd.median(t_signal) # apply 3rd order median filter and store the filtred signal in med_filtred
            
            if 'acc' in column: # test if the med_filtered signal is an acceleration signal 
                
                # the 2nd output DC_component is the gravity_acc
                # The 3rd one is the body_component which in this case the body_acc
                _,grav_acc,body_acc,_=frd.components_selection_one_signal(med_filtred,freq1,freq2) # apply components selection
                
                body_acc_jerk=frd.jerk_one_signal(body_acc)# apply the jerking function to body components only
                
                
                # store signal in time_sig_dataframe and delete the last value of each column 
                # jerked signal will have the original lenght-1(due to jerking)
                
                time_sig_df['t_body_'+column]=body_acc[:-1] # t_body_acc storing with the appropriate axis selected 
                #                                             from the column name
                
                time_sig_df['t_grav_'+column]= grav_acc[:-1] # t_grav_acc_storing with the appropriate axis selected 
                #                                              from the column name
                
                # store  t_body_acc_jerk signal with the appropriate axis selected from the column name
                time_sig_df['t_body_acc_jerk_'+column[-1]]=body_acc_jerk
            
            elif 'gyro' in column: # if the med_filtred signal is a gyro signal
                
                # The 3rd output of components_selection is the body_component which in this case the body_gyro component
                _,_,body_gyro,_=frd.components_selection_one_signal(med_filtred,freq1,freq2)  # apply components selection
                
                body_gyro_jerk=frd.jerk_one_signal(body_gyro) # apply the jerking function to body components only
                
                # store signal in time_sig_dataframe and delete the last value of each column 
                # jerked signal will have the original lenght-1(due to jerking)
                
                time_sig_df['t_body_gyro_'+column[-1]]=body_gyro[:-1] # t_body_acc storing with the appropriate axis selected 
                #                                                       from the column name
                
                time_sig_df['t_body_gyro_jerk_'+column[-1]]=body_gyro_jerk # t_grav_acc_storing with the appropriate axis 
                #                                                            selected from the column name
        
        
        # all 15 axial signals generated above are reordered to facilitate magnitudes signals generation
        new_columns_ordered=['t_body_acc_X','t_body_acc_Y','t_body_acc_Z',
                            't_grav_acc_X','t_grav_acc_Y','t_grav_acc_Z',
                            't_body_acc_jerk_X','t_body_acc_jerk_Y','t_body_acc_jerk_Z',
                            't_body_gyro_X','t_body_gyro_Y','t_body_gyro_Z',
                            't_body_gyro_jerk_X','t_body_gyro_jerk_Y','t_body_gyro_jerk_Z']
        
        # create new dataframe to order columns
        ordered_time_sig_df=pd.DataFrame()
        
        for col in new_columns_ordered: # iterate over each column in the new order
            ordered_time_sig_df[col]=time_sig_df[col] # store the column in the ordred dataframe
        
        # Generating magnitude signals
        for i in range(0,15,3): # iterating over each 3-axial signals
            
            mag_col_name=new_columns_ordered[i][:-1]+'mag'# Create the magnitude column name related to each 3-axial signals
            
            col0=np.array(ordered_time_sig_df[new_columns_ordered[i]]) # copy X_component
            col1=ordered_time_sig_df[new_columns_ordered[i+1]] # copy Y_component
            col2=ordered_time_sig_df[new_columns_ordered[i+2]] # copy Z_component
            
            mag_signal=frd.mag_3_signals(col0,col1,col2) # calculate magnitude of each signal[X,Y,Z]
            ordered_time_sig_df[mag_col_name]=mag_signal # store the signal_mag with its appropriate column name
        
        time_sig_dic[key]=ordered_time_sig_df # store the ordred_time_sig_df in time_sig_dic with the appropriate key

    # apply the sliding window type 1 to "time_sig dic"
    t_dic_win_type_I  = frd.Windowing_type_1(time_sig_dic,Labels_Data_Frame)

    # dictionaries includes f_windows obtained from t_windows type I and type II
    f_dic_win_type_I = {'f'+key[1:] : t_w1_df.pipe(frd.fast_fourier_transform) for key, t_w1_df in t_dic_win_type_I.items()}
    # apply datasets generation pipeline to time and frequency windows type I
    Dataset_type_I= fg.Dataset_Generation_PipeLine(t_dic_win_type_I,f_dic_win_type_I)
    print('The shape of Dataset type I is :',Dataset_type_I.shape) # shape of the dataset type I

if __name__ == "__main__":
    main()