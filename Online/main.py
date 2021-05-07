import numpy as np
import pandas as pd
import readData as rd
import filterRawData as frd
import featureGeneration as fg


############################## Constants #############################
path="./Data/exp01_user01.txt" # test data file

sampling_freq=50 # 50 Hz(hertz) is sampling frequency: the number of captured values of each axial signal per second.
freq1 = 0.3 # freq1=0.3 hertz [Hz] the cuttoff frequency between the DC compoenents [0,0.3] and the body components[0.3,20]hz
freq2 = 20  # freq2= 20 Hz the cuttoff frequcency between the body components[0.3,20] hz and the high frequency noise components [20,25] hz

#Size of chunk to remove noise and gravity
data_chunk_size=449

# Window size per original dataset
w_s=128

# Overlap per original dataset
overlap = 64

#data column names
raw_acc_columns=['acc_X','acc_Y','acc_Z']
raw_gyro_columns=['gyro_X','gyro_Y','gyro_Z']
column_names= raw_acc_columns + raw_gyro_columns

column_names_old=['t_body_acc_X','t_grav_acc_X','t_body_acc_jerk_X',
                  't_body_acc_Y','t_grav_acc_Y','t_body_acc_jerk_Y',
                  't_body_acc_Z','t_grav_acc_Z','t_body_acc_jerk_Z',
                  't_body_gyro_X','t_body_gyro_jerk_X','t_body_gyro_Y',
                  't_body_gyro_jerk_Y','t_body_gyro_Z','t_body_gyro_jerk_Z']

column_names_new=['t_body_acc_X','t_body_acc_Y','t_body_acc_Z',
                  't_grav_acc_X','t_grav_acc_Y','t_grav_acc_Z',
                  't_body_acc_jerk_X','t_body_acc_jerk_Y','t_body_acc_jerk_Z',
                  't_body_gyro_X','t_body_gyro_Y','t_body_gyro_Z',
                  't_body_gyro_jerk_X','t_body_gyro_jerk_Y','t_body_gyro_jerk_Z']


def main():
    
    # init vars
    raw_data = []
    samples_total = 0
    samples_chunk = 0
    blocks = 0
    # open the txt file with samples
    file=open(path,'r')
    # take each sample (3 acc components and 3 gyro components)
    for line in file:
        sample = rd.readSample(line)
        raw_data.append(sample)
        samples_total = samples_total + 1
        samples_chunk = samples_chunk + 1
        # when chunk of data is full perform denoising, generate windowed data and features
        if samples_chunk == data_chunk_size:
            samples_chunk = 0
            time_sig=np.empty((data_chunk_size-1,len(column_names_old)))
            if samples_total>data_chunk_size:
                data_chunk=np.array(raw_data[samples_total-data_chunk_size-overlap:samples_total-overlap])
            else:
                data_chunk=np.array(raw_data)
            c_i=0
            for col in range(np.shape(data_chunk)[2]):
                column=data_chunk[:,0,col]
                # perform median filtering
                med_filter_col=frd.median(column)
                if 'acc' in column_names[col]:
                    # perform component selection
                    _,grav_acc,body_acc,_ = frd.components_selection_one_signal(med_filter_col,sampling_freq,freq1,freq2)
                    
                    # compute jerked signal
                    body_acc_jerk=frd.jerk_one_signal(body_acc,sampling_freq)# apply the jerking function to body components only

                    # store signal in time_sig and delete the last value of each column 
                    # jerked signal will have the original lenght-1(due to jerking)
                    time_sig[:,c_i]=body_acc[:-1]
                    c_i+=1
                    
                    time_sig[:,c_i]= grav_acc[:-1]
                    c_i+=1

                    # store body_acc_jerk signal
                    time_sig[:,c_i]=body_acc_jerk
                    c_i+=1

                elif 'gyro' in column_names[col]:

                    # perform component selection
                    _,_,body_gyro,_ = frd.components_selection_one_signal(med_filter_col,sampling_freq,freq1,freq2)
                    
                    # compute jerk signal
                    body_gyro_jerk=frd.jerk_one_signal(body_gyro,sampling_freq)

                    # store gyro signal
                    time_sig[:,c_i]=body_gyro[:-1]
                    c_i+=1

                    # store body_gyro_jerk
                    time_sig[:,c_i]=body_gyro_jerk
                    c_i+=1

            # create new dataframe to order columns
            time_sig_df=pd.DataFrame()
            for col in column_names_new: # iterate over each column in the new order
                time_sig_df[col]=time_sig[:,column_names_old.index(col)] # store the column in the ordred dataframe
            
            # generate magnitude signals
            for i in range(0,15,3): # iterating over each 3-axial signals
        
                mag_col_name=column_names_new[i][:-1]+'mag'# create the magnitude column name related to each 3-axial signals
        
                col0=np.array(time_sig_df[column_names_new[i]]) # copy X_component
                col1=time_sig_df[column_names_new[i+1]] # copy Y_component
                col2=time_sig_df[column_names_new[i+2]] # copy Z_component
        
                mag_signal=frd.mag_3_signals(col0,col1,col2) # calculate magnitude of each signal[X,Y,Z]
                time_sig_df[mag_col_name]=mag_signal # store the signal_mag with its appropriate column name
            
            # apply sliding window
            t_W_dic=frd.Windowing(time_sig_df)
            
            # compute frequency domain signal
            f_W_dic={'f'+key[1:] : t_w1_df.pipe(frd.fast_fourier_transform) for key, t_w1_df in t_W_dic.items()}

            # conctenate all features names lists
            all_columns=fg.time_features_names()+fg.frequency_features_names()

            # apply datasets generation pipeline to time and frequency windows
            Dataset = fg.Dataset_Generation_PipeLine(t_W_dic,f_W_dic)
            print('The shape of Dataset is :',Dataset.shape) # shape of the dataset
            # Do stuff with dataset ...

if __name__ == "__main__":
    main()