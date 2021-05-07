import numpy as np
from numpy.fft import *

import scipy as sp
from scipy import fftpack
from scipy.signal import medfilt

import pandas as pd
import math

#median filter
def median(signal):# input: numpy array 1D (one column)
    array=np.array(signal)   
    #applying the median filter
    med_filtered=medfilt(array, kernel_size=3) # applying the median filter order3(kernel_size=3)
    return  med_filtered # return the med-filtered signal: numpy array 1D

# Function name: components_selection_one_signal

# Inputs: t_signal:1D numpy array (time domain signal); 

# Outputs: (total_component,t_DC_component , t_body_component, t_noise) 
#           type(1D array,1D array, 1D array)

# cases to discuss: if the t_signal is an acceleration signal then the t_DC_component is the gravity component [Grav_acc]
#                   if the t_signal is a gyro signal then the t_DC_component is not useful
# t_noise component is not useful
# if the t_signal is an acceleration signal then the t_body_component is the body's acceleration component [Body_acc]
# if the t_signal is a gyro signal then the t_body_component is the body's angular velocity component [Body_gyro]

def components_selection_one_signal(t_signal,sampling_freq,freq1,freq2):
    t_signal=np.array(t_signal)
    t_signal_length=len(t_signal) # number of points in a t_signal
    
    # the t_signal in frequency domain after applying fft
    f_signal=fft(t_signal) # 1D numpy array contains complex values (in C)
    
    # generate frequencies associated to f_signal complex values
    freqs=np.array(sp.fftpack.fftfreq(t_signal_length, d=1/float(sampling_freq))) # frequency values between [-25hz:+25hz]
    
    # DC_component: f_signal values having freq between [-0.3 hz to 0 hz] and from [0 hz to 0.3hz] 
    #                                                             (-0.3 and 0.3 are included)
    
    # noise components: f_signal values having freq between [-25 hz to 20 hz[ and from ] 20 hz to 25 hz] 
    #                                                               (-25 and 25 hz inculded 20hz and -20hz not included)
    
    # selecting body_component: f_signal values having freq between [-20 hz to -0.3 hz] and from [0.3 hz to 20 hz] 
    #                                                               (-0.3 and 0.3 not included , -20hz and 20 hz included)
    
    
    f_DC_signal=[] # DC_component in freq domain
    f_body_signal=[] # body component in freq domain numpy.append(a, a[0])
    f_noise_signal=[] # noise in freq domain
    
    for i in range(len(freqs)):# iterate over all available frequencies
        
        # selecting the frequency value
        freq=freqs[i]
        
        # selecting the f_signal value associated to freq
        value= f_signal[i]
        
        # Selecting DC_component values 
        if abs(freq)>freq1:# testing if freq is outside DC_component frequency ranges
            f_DC_signal.append(float(0)) # add 0 to  the  list if it was the case (the value should not be added)                                       
        else: # if freq is inside DC_component frequency ranges 
            f_DC_signal.append(value) # add f_signal value to f_DC_signal list
    
        # Selecting noise component values 
        if (abs(freq)<=freq2):# testing if freq is outside noise frequency ranges 
            f_noise_signal.append(float(0)) # # add 0 to  f_noise_signal list if it was the case 
        else:# if freq is inside noise frequency ranges 
            f_noise_signal.append(value) # add f_signal value to f_noise_signal

        # Selecting body_component values 
        if (abs(freq)<=freq1 or abs(freq)>freq2):# testing if freq is outside Body_component frequency ranges
            f_body_signal.append(float(0))# add 0 to  f_body_signal list
        else:# if freq is inside Body_component frequency ranges
            f_body_signal.append(value) # add f_signal value to f_body_signal list
    
    ################### Inverse the transformation of signals in freq domain ########################
    # applying the inverse fft(ifft) to signals in freq domain and put them in float format
    t_DC_component= ifft(np.array(f_DC_signal)).real
    t_body_component= ifft(np.array(f_body_signal)).real
    t_noise=ifft(np.array(f_noise_signal)).real
    
    total_component=t_signal-t_noise # extracting the total component(filtered from noise) 
                                     #  by substracting noise from t_signal (the original signal).
    
    # return outputs mentioned earlier
    return (total_component,t_DC_component,t_body_component,t_noise)


def jerk_one_signal(signal, sampling_freq):
    dt=1.0/sampling_freq 
    return np.array([(signal[i+1]-signal[i])/dt for i in range(len(signal)-1)])

def mag_3_signals(x,y,z):# magnitude function redefintion
    return np.array([math.sqrt((x[i]**2+y[i]**2+z[i]**2)) for i in range(len(x))])

# it add '0's to the left of the input until the new lenght is equal to 2
def normalize5(number): 
    stre=str(number)
    if len(stre)<5:
        l=len(stre)
        for i in range(0,5-l):
            stre="0"+stre
    return stre

def Windowing(time_sig_df):   
    
    window_ID=0 # window unique id
    t_dic_win_type_I={} # output dic

    rows=time_sig_df.shape[0]
    columns = time_sig_df.columns
    # from the cursor we copy a window that has 128 rows
    # the cursor step is 64 data point (50% of overlap) : each time it will be shifted by 64 rows
    for cursor in range(0,rows-127,64):

        

        # end_point: cursor(the first index in the window) + 128
        end_point=cursor+128 # window end row

        # selecting window data points convert them to numpy array to delete rows index
        data=np.array(time_sig_df.iloc[cursor:end_point])

        # converting numpy array to a dataframe with the same column names
        window=pd.DataFrame(data=data,columns=columns)

        # creating the window
        key='t_W'+normalize5(window_ID)
        t_dic_win_type_I[key]=window

        # incrementing the windowID by 1
        window_ID=window_ID+1
        
    return t_dic_win_type_I # return a dictionary including time domain windows type I

# Inputs: time signal 1D array
# Output: amplitude of fft components 1D array having the same lenght as the Input
def fast_fourier_transform_one_signal(t_signal):
    # apply fast fourrier transform to the t_signal
    complex_f_signal= fftpack.fft(t_signal)
    #compute the amplitude each complex number
    amplitude_f_signal=np.abs(complex_f_signal)
    # return the amplitude
    return amplitude_f_signal

# Inputs: A DataFrame with 20 time signal (20 columns) gravity columns(4) won't be transformed
# Outputs: A DataFrame with 16 frequency signal (16 columns)
def fast_fourier_transform(t_window):
    
    f_window=pd.DataFrame() # create an empty dataframe will include frequency domain signals of window
    
    for column in t_window.columns: # iterating over time domain window columns(signals)
        
        if 'grav' not in column: # verify if time domain signal is not related to gravity components
            
            t_signal=np.array(t_window[column]) # convert the column to a 1D numpy array
           
            f_signal= np.apply_along_axis(fast_fourier_transform_one_signal,0,t_signal) # apply the function defined above to the column
            f_window["f_"+column[2:]]=f_signal # storing the frequency signal in f_window with an appropriate column name
    
    return f_window # return the frequency domain window