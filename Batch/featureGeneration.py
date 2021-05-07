# Importing Scipy 
import scipy as sp
# df is dataframe contains 3 columns (3 axial signals X,Y,Z)
# Importing numpy 
import numpy as np

# Importing Pandas Library 
import pandas as pd

import math

# mean
def mean_axial(df):
    array=np.array(df) # convert dataframe into 2D numpy array for efficiency
    mean_vector = list(array.mean(axis=0)) # calculate the mean value of each column
    return mean_vector # return mean vetor
# std
def std_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    std_vector = list(array.std(axis=0))# calculate the standard deviation value of each column
    return std_vector

# mad
from statsmodels.robust import mad as median_deviation # import the median deviation function
def mad_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    mad_vector = list(median_deviation(array,axis=0)) # calculate the median deviation value of each column
    return mad_vector

# max

def max_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    max_vector=list(array.max(axis=0))# calculate the max value of each column
    return max_vector
# min
def min_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    min_vector=list(array.min(axis=0))# calculate the min value of each column
    return min_vector
# IQR
from scipy.stats import iqr as IQR # import interquartile range function (Q3(column)-Q1(column))
def IQR_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    IQR_vector=list(np.apply_along_axis(IQR,0,array))# calculate the inter quartile range value of each column
    return IQR_vector


# Entropy
from scipy.stats import entropy # import the entropy function
def entropy_axial(df):
    array=np.array(df)# convert dataframe into 2D numpy array for efficiency
    entropy_vector=list(np.apply_along_axis(entropy,0,abs(array)))# calculate the entropy value of each column
    return entropy_vector

# mag column : is one column contains one mag signal values
# same features mentioned above were calculated for each column

# mean
def mean_mag(mag_column):
    array=np.array(mag_column)
    mean_value = float(array.mean())
    return mean_value

# std: standard deviation of mag column
def std_mag(mag_column):
    array=np.array(mag_column)
    std_value = float(array.std()) # std value 
    return std_value

# mad: median deviation
def mad_mag(mag_column):
    array=np.array(mag_column)
    mad_value = float(median_deviation(array))# median deviation value of mag_column
    return mad_value

# max
def max_mag(mag_column):
    array=np.array(mag_column)
    max_value=float(array.max()) # max value 
    return max_value
# min
def min_mag(mag_column):
    array=np.array(mag_column)
    min_value= float(array.min()) # min value
    return min_value

# IQR
def IQR_mag(mag_column):
    array=np.array(mag_column)
    IQR_value=float(IQR(array))# Q3(column)-Q1(column)
    return IQR_value

# Entropy
def entropy_mag(mag_column):
    array=np.array(mag_column)
    entropy_value=float(entropy(array)) # entropy signal
    return entropy_value


# Functions used to generate time axial features

# df is dataframe contains 3 columns (3 axial signals X,Y,Z)
# sma
def t_sma_axial(df):
    array=np.array(df)
    sma_axial=float(abs(array).sum())/float(3) # sum of areas under each signal
    return sma_axial # return sma value

# energy
def t_energy_axial(df):
    array=np.array(df)
    energy_vector=list((array**2).sum(axis=0)) # energy value of each df column
    return energy_vector # return energy vector energy_X,energy_Y,energy_Z

# AR vector (auto regression coefficients from 1 to 4)

# define the arbugr function
#auto regression coefficients with using burg method with order from 1 to 4
from spectrum import *

##############################################################################################
# I took this function as it is from this link ------>    https://github.com/faroit/freezefx/blob/master/fastburg.py
# This fucntion and the original function arburg in the library spectrum generate the same first 3 coefficients 
#for all windows the original burg method is low and for some windows it cannot generate all 4th coefficients 

def _arburg2(X, order):
    """This version is 10 times faster than arburg, but the output rho is not correct.
    returns [1 a0,a1, an-1]
    """
    x = numpy.array(X)
    N = len(x)

    if order == 0.:
        raise ValueError("order must be > 0")

    # Initialisation
    # ------ rho, den
    rho = sum(abs(x)**2.) / N  # Eq 8.21 [Marple]_
    den = rho * 2. * N

    # ------ backward and forward errors
    ef = numpy.zeros(N, dtype=complex)
    eb = numpy.zeros(N, dtype=complex)
    for j in range(0, N):  # eq 8.11
        ef[j] = x[j]
        eb[j] = x[j]

    # AR order to be stored
    a = numpy.zeros(1, dtype=complex)
    a[0] = 1
    # ---- rflection coeff to be stored
    ref = numpy.zeros(order, dtype=complex)

    E = numpy.zeros(order+1)
    E[0] = rho

    for m in range(0, order):
        # print m
        # Calculate the next order reflection (parcor) coefficient
        efp = ef[1:]
        ebp = eb[0:-1]
        # print efp, ebp
        num = -2. * numpy.dot(ebp.conj().transpose(), efp)
        den = numpy.dot(efp.conj().transpose(),  efp)
        den += numpy.dot(ebp,  ebp.conj().transpose())
        ref[m] = num / den

        # Update the forward and backward prediction errors
        ef = efp + ref[m] * ebp
        eb = ebp + ref[m].conj().transpose() * efp

        # Update the AR coeff.
        a.resize(len(a)+1, refcheck=False)
        a = a + ref[m] * numpy.flipud(a).conjugate()

        # Update the prediction error
        E[m+1] = numpy.real((1 - ref[m].conj().transpose() * ref[m])) * E[m]
        # print 'REF', ref, num, den
    return a, E[-1], ref

#################################################################################################################

# to generate arburg (order 4) coefficents for 3 columns [X,Y,Z]
def t_arburg_axial(df):
    # converting signals to 1D numpy arrays for efficiency
    array_X=np.array(df[df.columns[0]])
    array_Y=np.array(df[df.columns[1]])
    array_Z=np.array(df[df.columns[2]])
    
    AR_X = list(_arburg2(array_X,4)[0][1:].real) # list contains real parts of all 4th coefficients generated from signal_X
    AR_Y = list(_arburg2(array_Y,4)[0][1:].real) # list contains real parts of all 4th coefficients generated from signal_Y
    AR_Z = list(_arburg2(array_Z,4)[0][1:].real) # list contains real parts of all 4th coefficients generated from signal_Z
    
    # selecting [AR1 AR2 AR3 AR4] real components for each axis concatenate them in one vector
    AR_vector= AR_X + AR_Y+ AR_Z
    
    
    # AR_vector contains 12 values 4values per each axis 
    return AR_vector


from scipy.stats import pearsonr
def t_corr_axial(df): # it returns 3 correlation features per each 3-axial signals in  time_window
    
    array=np.array(df)
    
    Corr_X_Y=float(pearsonr(array[:,0],array[:,1])[0]) # correlation value between signal_X and signal_Y
    Corr_X_Z=float(pearsonr(array[:,0],array[:,2])[0]) # correlation value between signal_X and signal_Z
    Corr_Y_Z=float(pearsonr(array[:,1],array[:,2])[0]) # correlation value between signal_Y and signal_Z
    
    corr_vector =[Corr_X_Y, Corr_X_Z, Corr_Y_Z] # put correlation values in list
    
    return corr_vector

def t_axial_features_generation(t_window):
    
    # select axial columns : the first 15 columns
    axial_columns=t_window.columns[0:15]
    
    # select axial columns in a dataframe
    axial_df=t_window[axial_columns]
    
    ## a list will contain all axial features values resulted from applying: 
    #  common axial features functions and time axial features functions to all time domain signals in t_window
    t_axial_features=[]
    for col in range(0,15,3):
        df=axial_df[axial_columns[col:col+3]] # select each group of 3-axial signal: signal_name[X,Y,Z]
        
        # apply all common axial features functions and time axial features functions to each 3-axial signals dataframe
        mean_vector   = mean_axial(df) # 3values
        std_vector    = std_axial(df) # 3 values
        mad_vector    = mad_axial(df)# 3 values
        max_vector    = max_axial(df)# 3 values
        min_vector    = min_axial(df)# 3 values
        sma_value     = t_sma_axial(df)# 1 value
        energy_vector = t_energy_axial(df)# 3 values
        IQR_vector    = IQR_axial(df)# 3 values
        entropy_vector= entropy_axial(df)# 3 values
        AR_vector     = t_arburg_axial(df)# 3 values
        corr_vector   = t_corr_axial(df)# 3 values
        
        # 40 value per each 3-axial signals
        t_3axial_vector= mean_vector + std_vector + mad_vector + max_vector + min_vector + [sma_value] + energy_vector + IQR_vector + entropy_vector + AR_vector + corr_vector
        
        # append these features to the global list of features
        t_axial_features= t_axial_features+ t_3axial_vector
    
    # t_axial_features contains 200 values = 40 value per each 3axial x 5 tri-axial-signals[X,Y,Z]
    return t_axial_features

    # Functions used to generate time magnitude features




# sma: signal magnitude area
def t_sma_mag(mag_column):
    array=np.array(mag_column)
    sma_mag=float(abs(array).sum())# signal magnitude area of one mag column
    return sma_mag

# energy
def t_energy_mag(mag_column):
    array=np.array(mag_column)
    energy_value=float((array**2).sum()) # energy of the mag signal
    return energy_value



# arburg: auto regression coefficients using the burg method
def t_arburg_mag(mag_column):
    
    array = np.array(mag_column)
    
    AR_vector= list(_arburg2(array,4)[0][1:].real) # AR1, AR2, AR3, AR4 of the mag column
    #print(AR_vector)
    return AR_vector

def t_mag_features_generation(t_window):
    
    # select mag columns : the last 5 columns in a time domain window
    
    mag_columns=t_window.columns[15:] # mag columns' names
    mag_columns=t_window[mag_columns] # mag data frame
    
    t_mag_features=[] # a global list will contain all time domain magnitude features
    
    for col in mag_columns: # iterate throw each mag column
        
        mean_value   = mean_mag(mag_columns[col]) # 1 value
        std_value    = std_mag(mag_columns[col])# 1 value
        mad_value    = mad_mag(mag_columns[col])# 1 value
        max_value    = max_mag(mag_columns[col])# 1 value
        min_value    = min_mag(mag_columns[col])# 1 value
        sma_value    = t_sma_mag(mag_columns[col])# 1 value
        energy_value = t_energy_mag(mag_columns[col])# 1 value
        IQR_value    = IQR_mag(mag_columns[col])# 1 value
        entropy_value= entropy_mag(mag_columns[col])# 1 value
        AR_vector    = t_arburg_mag(mag_columns[col])# 1 value
        
        # 13 value per each t_mag_column
        col_mag_values = [mean_value, std_value, mad_value, max_value, min_value, sma_value, 
                          energy_value,IQR_value, entropy_value]+ AR_vector
        
        # col_mag_values will be added to the global list
        t_mag_features= t_mag_features+ col_mag_values
    
    # t_mag_features contains 65 values = 13 values (per each t_mag_column) x 5 (t_mag_columns)
    return t_mag_features

def time_features_names():
# Generating time feature names

# time domain axial signals' names
    t_axis_signals=[['t_body_acc_X','t_body_acc_Y','t_body_acc_Z'],
                    ['t_grav_acc_X','t_grav_acc_Y','t_grav_acc_Z'],
                    ['t_body_acc_jerk_X','t_body_acc_jerk_Y','t_body_acc_jerk_Z'],    

                    ['t_body_gyro_X','t_body_gyro_Y','t_body_gyro_Z'],
            ['t_body_gyro_Jerk_X','t_body_gyro_Jerk_Y','t_body_gyro_Jerk_Z'],]
    
    # time domain magnitude signals' names
    magnitude_signals=['t_body_acc_Mag','t_grav_acc_Mag','t_body_acc_jerk_Mag','t_body_gyro_Mag','t_body_gyro_Jerk_Mag']

    # functions' names:
    t_one_input_features_name1=['_mean()','_std()','_mad()','_max()','_min()']

    t_one_input_features_name2=['_energy()','_iqr()','_entropy()']

    t_one_input_features_name3=['_AR1()','_AR2()','_AR3()','_AR4()']

    correlation_columns=['_Corr(X,Y)','_Corr(X,Z)','_Corr(Y,Z)']

    

    features=[]# Empty list : it will contain all time domain features' names
    
    for columns in t_axis_signals: # iterate throw  each group of 3-axial signals'
        
        for feature in t_one_input_features_name1: # iterate throw the first list of functions names
            
            for column in columns: # iterate throw each axial signal in that group
                
                newcolumn=column[:-2]+feature+column[-2:] # build the feature name
                features.append(newcolumn) # add it to the global list
        
        sma_column=column[:-2]+'_sma()' # build the feature name sma related to that group
        features.append(sma_column) # add the feature to the list
        
        for feature in t_one_input_features_name2: # same process for the second list of features functions
            for column in columns:
                newcolumn=column[:-2]+feature+column[-2:]
                features.append(newcolumn)
        
        for column in columns:# same process for the third list of features functions
            for feature in t_one_input_features_name3:
                newcolumn=column[0:-2]+feature+column[-2:]
                features.append(newcolumn)
        
        for feature in correlation_columns: # adding correlations features
            newcolumn=column[0:-2]+feature
            features.append(newcolumn)

    for columns in magnitude_signals: # iterate throw time domain magnitude column names

        # build feature names related to that column
        #list 1
        for feature in t_one_input_features_name1:
            newcolumn=columns+feature
            features.append(newcolumn)
        # sma feature name
        sma_column=columns+'_sma()'
        features.append(sma_column)
        
        # list 2
        for feature in t_one_input_features_name2: 
            newcolumn=columns+feature
            features.append(newcolumn)
        
        # list 3
        for feature in t_one_input_features_name3:
            newcolumn=columns+feature
            features.append(newcolumn)
    ###########################################################################################################
    time_list_features=features
    
    return time_list_features # return all time domain features' names

    # Functions used to generate frequency axial features
# each df here is dataframe contains 3 columns (3 axial frequency domain signals X,Y,Z)
# signals were obtained from frequency domain windows
# sma
def f_sma_axial(df):
    
    array=np.array(df)
    sma_value=float((abs(array)/math.sqrt(128)).sum())/float(3) # sma value of 3-axial f_signals
    
    return sma_value

# energy
def f_energy_axial(df):
    
    array=np.array(df)
    
    # spectral energy vector
    energy_vector=list((array**2).sum(axis=0)/float(len(array))) # energy of: f_signalX,f_signalY, f_signalZ
    
    return energy_vector # enrgy veactor=[energy(signal_X),energy(signal_Y),energy(signal_Z)]


####### Max Inds and Mean_Freq Functions#######################################
# built frequencies list (each column contain 128 value)
# duration between each two successive captures is 0.02 s= 1/50hz
freqs=sp.fftpack.fftfreq(128, d=0.02) 
                                

# max_Inds
def f_max_Inds_axial(df):
    array=np.array(df)
    max_Inds_X =freqs[array[1:65,0].argmax()+1] # return the frequency related to max value of f_signal X
    max_Inds_Y =freqs[array[1:65,1].argmax()+1] # return the frequency related to max value of f_signal Y
    max_Inds_Z =freqs[array[1:65,2].argmax()+1] # return the frequency related to max value of f_signal Z
    max_Inds_vector= [max_Inds_X,max_Inds_Y,max_Inds_Z]# put those frequencies in a list
    return max_Inds_vector

# mean freq()
def f_mean_Freq_axial(df):
    array=np.array(df)
    
    # sum of( freq_i * f_signal[i])/ sum of signal[i]
    mean_freq_X = np.dot(freqs,array[:,0]).sum() / float(array[:,0].sum()) #  frequencies weighted sum using f_signalX
    mean_freq_Y = np.dot(freqs,array[:,1]).sum() / float(array[:,1].sum()) #  frequencies weighted sum using f_signalY 
    mean_freq_Z = np.dot(freqs,array[:,2]).sum() / float(array[:,2].sum()) #  frequencies weighted sum using f_signalZ
    
    mean_freq_vector=[mean_freq_X,mean_freq_Y,mean_freq_Z] # vector contain mean frequencies[X,Y,Z]
    
    return  mean_freq_vector

###################################################################################

########## Skewness & Kurtosis Functions #######################################
from scipy.stats import kurtosis       # kurtosis function
from scipy.stats import skew           # skewness function
    
def f_skewness_and_kurtosis_axial(df):
    array=np.array(df)
    
    skew_X= skew(array[:,0])  # skewness value of signal X
    kur_X= kurtosis(array[:,0])  # kurtosis value of signal X
    
    skew_Y= skew(array[:,1]) # skewness value of signal Y
    kur_Y= kurtosis(array[:,1])# kurtosis value of signal Y
    
    skew_Z= skew(array[:,2])# skewness value of signal Z
    kur_Z= kurtosis(array[:,2])# kurtosis value of signal Z
    
    skew_kur_3axial_vector=[skew_X,kur_X,skew_Y,kur_Y,skew_Z,kur_Z] # return the list
    
    return skew_kur_3axial_vector
##################################################################################


#################### Bands Energy FUNCTIONS ########################

# bands energy levels (start row,end_row) end row not included 
B1=[(1,9),(9,17),(17,25),(25,33),(33,41),(41,49),(49,57),(57,65)] 
B2=[(1,17),(17,31),(31,49),(49,65)]
B3=[(1,25),(25,49)]

def f_one_band_energy(f_signal,band): # f_signal is one column in frequency axial signals in f_window
    # band: is one tuple in B1 ,B2 or B3 
    f_signal_bounded = f_signal[band[0]:band[1]] # select f_signal components included in the band
    energy_value=float((f_signal_bounded**2).sum()/float(len(f_signal_bounded))) # energy value of that band
    return energy_value

def f_all_bands_energy_axial(df): # df is dataframe contain 3 columns (3-axial f_signals [X,Y,Z])
    
    E_3_axis =[]
    
    array=np.array(df)
    for i in range(0,3): # iterate throw signals
        E1=[ f_one_band_energy( array,( B1 [j][0], B1 [j][1]) ) for j in range(len(B1))] # energy bands1 values of f_signal
        E2=[ f_one_band_energy( array,( B2 [j][0], B2 [j][1]) ) for j in range(len(B2))]# energy bands2 values of f_signal
        E3=[ f_one_band_energy( array,( B3 [j][0], B3 [j][1]) ) for j in range(len(B3))]# energy bands3 values of f_signal
    
        E_one_axis = E1+E2+E3 # list of energy bands values of one f_signal
        
        E_3_axis= E_3_axis + E_one_axis # add values to the global list
    
    return E_3_axis

        
    
    axial_columns=f_window.columns[0:12] # select frequency axial column names
    axial_df=f_window[axial_columns] # select frequency axial signals in one dataframe
    f_all_axial_features=[] # a global list will contain all frequency axial features values
    
    
    
    for col in range(0,12,3):# iterate throw each group of frequency axial signals in a window
        
        df=axial_df[axial_columns[col:col+3]]  # select each group of 3-axial signals
      
        # mean
        mean_vector                  = mean_axial(df) # 3 values
        # std
        std_vector                   = std_axial(df) # 3 values
        # mad
        mad_vector                   = mad_axial(df) # 3 values
        # max
        max_vector                   = max_axial(df) # 3 values
        # min
        min_vector                   = min_axial(df) # 3 values
        # sma
        sma_value                    = f_sma_axial(df)
        # energy
        energy_vector                = f_energy_axial(df)# 3 values
        # IQR
        IQR_vector                   = IQR_axial(df) # 3 values
        # entropy
        entropy_vector               = entropy_axial(df) # 3 values
        # max_inds
        max_inds_vector              = f_max_Inds_axial(df)# 3 values
        # mean_Freq
        mean_Freq_vector             = f_mean_Freq_axial(df)# 3 values
        # skewness and kurtosis
        skewness_and_kurtosis_vector = f_skewness_and_kurtosis_axial(df)# 6 values
        # bands energy
        bands_energy_vector          = f_all_bands_energy_axial(df) # 42 values

        # append all values of each 3-axial signals in a list
        f_3axial_features = mean_vector +std_vector + mad_vector + max_vector + min_vector + [sma_value] + energy_vector + IQR_vector + entropy_vector + max_inds_vector + mean_Freq_vector + skewness_and_kurtosis_vector + bands_energy_vector

        f_all_axial_features = f_all_axial_features+ f_3axial_features # add features to the global list
        
    return f_all_axial_features

    # Functions used to generate frequency magnitude features

# sma
def f_sma_mag(mag_column):
    
    array=np.array(mag_column)
    sma_value=float((abs(array)/math.sqrt(len(mag_column))).sum()) # sma of one mag f_signals
    
    return sma_value

# energy
def f_energy_mag(mag_column):
    
    array=np.array(mag_column)
    # spectral energy value
    energy_value=float((array**2).sum()/float(len(array))) # energy value of one mag f_signals
    return energy_value


####### Max Inds and Mean_Freq Functions#######################################


# max_Inds
def f_max_Inds_mag(mag_column):
    
    array=np.array(mag_column)
    
    max_Inds_value =float(freqs[array[1:65].argmax()+1]) # freq value related with max component
    
    return max_Inds_value

# mean freq()
def f_mean_Freq_mag(mag_column):
    
    array=np.array(mag_column)
    
    mean_freq_value = float(np.dot(freqs,array).sum() / float(array.sum())) # weighted sum of one mag f_signal
    
    return  mean_freq_value

###################################################################################

########## Skewness & Kurtosis Functions #######################################

from scipy.stats import skew           # skewness
def f_skewness_mag(mag_column):
    
    array=np.array(mag_column)
    skew_value     = float(skew(array)) # skewness value of one mag f_signal
    return skew_value



from scipy.stats import kurtosis       # kurtosis
def f_kurtosis_mag(mag_column):
    array=np.array(mag_column)
    kurtosis_value = float(kurtosis(array)) # kurotosis value of on mag f_signal

    return kurtosis_value
##################################################################################

def f_mag_features_generation(f_window):
    
    # select frequnecy mag columns : the last 4 columns in f_window
    mag_columns=f_window.columns[-4:]
    mag_columns=f_window[mag_columns]
    
    f_mag_features=[]
    for col in mag_columns: # iterate throw each mag column in f_window
        
        # calculate common mag features and frequency mag features for each column
        mean_value   = mean_mag(mag_columns[col])
        std_value    = std_mag(mag_columns[col])
        mad_value    = mad_mag(mag_columns[col])
        max_value    = max_mag(mag_columns[col])
        min_value    = min_mag(mag_columns[col])
        sma_value    = f_sma_mag(mag_columns[col])
        energy_value = f_energy_mag(mag_columns[col])
        IQR_value    = IQR_mag(mag_columns[col])
        entropy_value= entropy_mag(mag_columns[col])
        max_Inds_value=f_max_Inds_mag(mag_columns[col])
        mean_Freq_value= f_mean_Freq_mag (mag_columns[col])
        skewness_value=  f_skewness_mag(mag_columns[col])
        kurtosis_value = f_kurtosis_mag(mag_columns[col])
        # 13 value per each t_mag_column
        col_mag_values = [mean_value, std_value, mad_value, max_value, 
                          min_value, sma_value, energy_value,IQR_value, 
                          entropy_value, max_Inds_value, mean_Freq_value,
                          skewness_value, kurtosis_value ]
        
        
        f_mag_features= f_mag_features+ col_mag_values # append feature values of one mag column to the global list
    
    # f_mag_features contains 65 values = 13 value (per each t_mag_column) x 4 (f_mag_columns)
    return f_mag_features

def f_axial_features_generation(f_window):
    
    
    axial_columns=f_window.columns[0:12] # select frequency axial column names
    axial_df=f_window[axial_columns] # select frequency axial signals in one dataframe
    f_all_axial_features=[] # a global list will contain all frequency axial features values
    
    
    
    for col in range(0,12,3):# iterate throw each group of frequency axial signals in a window
        
        df=axial_df[axial_columns[col:col+3]]  # select each group of 3-axial signals
      
        # mean
        mean_vector                  = mean_axial(df) # 3 values
        # std
        std_vector                   = std_axial(df) # 3 values
        # mad
        mad_vector                   = mad_axial(df) # 3 values
        # max
        max_vector                   = max_axial(df) # 3 values
        # min
        min_vector                   = min_axial(df) # 3 values
        # sma
        sma_value                    = f_sma_axial(df)
        # energy
        energy_vector                = f_energy_axial(df)# 3 values
        # IQR
        IQR_vector                   = IQR_axial(df) # 3 values
        # entropy
        entropy_vector               = entropy_axial(df) # 3 values
        # max_inds
        max_inds_vector              = f_max_Inds_axial(df)# 3 values
        # mean_Freq
        mean_Freq_vector             = f_mean_Freq_axial(df)# 3 values
        # skewness and kurtosis
        skewness_and_kurtosis_vector = f_skewness_and_kurtosis_axial(df)# 6 values
        # bands energy
        bands_energy_vector          = f_all_bands_energy_axial(df) # 42 values

        # append all values of each 3-axial signals in a list
        f_3axial_features = mean_vector +std_vector + mad_vector + max_vector + min_vector + [sma_value] + energy_vector + IQR_vector + entropy_vector + max_inds_vector + mean_Freq_vector + skewness_and_kurtosis_vector + bands_energy_vector

        f_all_axial_features = f_all_axial_features+ f_3axial_features # add features to the global list
        
    return f_all_axial_features

def frequency_features_names():
    #Generating Frequency feature names
    
    # frequency axial signal names 
    axial_signals=[
                    ['f_body_acc_X','f_body_acc_Y','f_body_acc_Z'],
                    ['f_body_acc_Jerk_X','f_body_acc_Jerk_Y','f_body_acc_Jerk_Z'],
                    ['f_body_gyro_X','f_body_gyro_Y','f_body_gyro_Z'],
                    ['f_body_gyro_Jerk_X','f_body_gyro_Jerk_Y','f_body_gyro_Jerk_Z'],
                ]

    # frequency magnitude signals
    mag_signals=['f_body_acc_Mag','f_body_acc_Jerk_Mag','f_body_gyro_Mag','f_body_gyro_Jerk_Mag']


    # features functions names will be applied to f_signals
    f_one_input_features_name1=['_mean()','_std()','_mad()','_max()','_min()']

    f_one_input_features_name2=['_energy()','_iqr()','_entropy()','_maxInd()','_meanFreq()']

    f_one_input_features_name3= ['_skewness()','_kurtosis()']

    f_one_input_features_name4=[
                                '_BE[1-8]','_BE[9-16]','_BE[17-24]','_BE[25-32]',
                                '_BE[33-40]','_BE[41-48]','_BE[49-56]','_BE[57-64]',
                                '_BE[1-16]','_BE[17-32]','_BE[33-48]','_BE[49-64]',
                                '_BE[1-24]','_BE[25-48]'
                            ]
    
    frequency_features_names=[] # global list of frequency features
    
    for columns in axial_signals: # iterate throw each group of 3-axial signals
        
        # iterate throw the first list of features
        for feature in f_one_input_features_name1: 
            for column in columns:# iterate throw each signal name of that group
                newcolumn=column[:-2]+feature+column[-2:] # build the full feature name
                frequency_features_names.append(newcolumn) # add the feature name to the global list
        
        # sma feature name
        sma_column=column[:-2]+'_sma()'
        frequency_features_names.append(sma_column)

        # iterate throw the first list of features
        for feature in f_one_input_features_name2:
            for column in columns:
                newcolumn=column[:-2]+feature+column[-2:]
                frequency_features_names.append(newcolumn)
        
        # iterate throw each signal name of that group
        for column in columns:
            for feature in f_one_input_features_name3: # iterate throw [skewness ,kurtosis]
                newcolumn=column[:-2]+feature+column[-2:] # build full feature name
                frequency_features_names.append(newcolumn) # append full feature names
        
        # same process above will be applied to list number 4
        for column in columns:
            for feature in f_one_input_features_name4:
                newcolumn=column[:-2]+feature+column[-2:]
                frequency_features_names.append(newcolumn)

    #################################################################################################################
    # generate frequency mag features names
    for column in mag_signals:# iterate throw each frequency mag signal name
        for feature in f_one_input_features_name1:# iterate throw the first list of features functions names
            frequency_features_names.append(column+feature) # build the full feature name and add it to the global list

        sma_column=column+'_sma()' # build the sma full feature name
        frequency_features_names.append(sma_column) # add it to the global list

        for feature in f_one_input_features_name2:# iterate throw the second list of features functions names
            frequency_features_names.append(column+feature)# build the full feature name and add it to the global list
        
        for feature in f_one_input_features_name3:# iterate throw the third list of features functions names
            frequency_features_names.append(column+feature)# build the full feature name and add it to the global list
    ####################################################################################################################
    
    return frequency_features_names

from math import acos # inverse of cosinus function
from math import sqrt # square root function

def magnitude_vector(vector3D): # vector[X,Y,Z]
    return sqrt((vector3D**2).sum()) # eulidian norm of that vector

###########angle between two vectors in radian ###############
def angle(vector1, vector2):
    vector1_mag=magnitude_vector(vector1) # euclidian norm of V1
    vector2_mag=magnitude_vector(vector2) # euclidian norm of V2
   
    scalar_product=np.dot(vector1,vector2) # scalar product of vector 1 and Vector 2
    cos_angle=scalar_product/float(vector1_mag*vector2_mag) # the cosinus value of the angle between V1 and V2
    
    # just in case some values were added automatically
    if cos_angle>1:
        cos_angle=1
    elif cos_angle<-1:
        cos_angle=-1
    
    angle_value=float(acos(cos_angle)) # the angle value in radian
    return angle_value # in radian.

################## angle_features ############################
def angle_features(t_window): # it returns 7 angles per window
    angles_list=[]# global list of angles values
    
    # mean value of each column t_body_acc[X,Y,Z]
    V2_columns=['t_grav_acc_X','t_grav_acc_Y','t_grav_acc_Z']
    V2_Vector=np.array(t_window[V2_columns].mean()) # mean values
    
    # angle 0: angle between (t_body_acc[X.mean,Y.mean,Z.mean], t_gravity[X.mean,Y.mean,Z.mean])
    V1_columns=['t_body_acc_X','t_body_acc_Y','t_body_acc_Z']
    V1_Vector=np.array(t_window[V1_columns].mean()) # mean values of t_body_acc[X,Y,Z]
    angles_list.append(angle(V1_Vector, V2_Vector)) # angle between the vectors added to the global list
    
    # same process is applied to ither signals
    # angle 1: (t_body_acc_jerk[X.mean,Y.mean,Z.mean],t_gravity[X.mean,Y.mean,Z.mean]
    V1_columns=['t_body_acc_jerk_X','t_body_acc_jerk_Y','t_body_acc_jerk_Z']
    V1_Vector=np.array(t_window[V1_columns].mean())
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # angle 2: (t_body_gyro[X.mean,Y.mean,Z.mean],t_gravity[X.mean,Y.mean,Z.mean]
    V1_columns=['t_body_gyro_X','t_body_gyro_Y','t_body_gyro_Z']
    V1_Vector=np.array(t_window[V1_columns].mean())
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # angle 3: (t_body_gyro_jerk[X.mean,Y.mean,Z.mean],t_gravity[X.mean,Y.mean,Z.mean]
    V1_columns=['t_body_gyro_jerk_X','t_body_gyro_jerk_Y','t_body_gyro_jerk_Z']
    V1_Vector=np.array(t_window[V1_columns].mean())
    angles_list.append(angle(V1_Vector, V2_Vector))
    #################################################################################
    
    # V1 vector in this case is the X axis itself [1,0,0]
    # angle 4: ([X_axis],t_gravity[X.mean,Y.mean,Z.mean])   
    V1_Vector=np.array([1,0,0])
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # V1 vector in this case is the Y axis itself [0,1,0]
    # angle 5: ([Y_acc_axis],t_gravity[X.mean,Y.mean,Z.mean]) 
    V1_Vector=np.array([0,1,0])
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    # V1 vector in this case is the Z axis itself [0,0,1]
    # angle 6: ([Z_acc_axis],t_gravity[X.mean,Y.mean,Z.mean])
    V1_Vector=np.array([0,0,1])
    angles_list.append(angle(V1_Vector, V2_Vector))
    
    return angles_list

def Dataset_Generation_PipeLine(t_dic,f_dic):
    # t_dic is a dic contains time domain windows
    # f_dic is a dic contains frequency domain windows
    # f_dic should be the result of applying fft to t_dic
    angle_columns=['angle0()','angle1()','angle2()','angle3()','angle4()','angle5()','angle6()']
    all_columns=time_features_names()+frequency_features_names()+angle_columns+['activity_Id','user_Id']
    final_Dataset=pd.DataFrame(data=[],columns= all_columns) # build an empty dataframe to append rows
    
    for i in range(len(t_dic)): # iterate throw each window

        # t_window and f_window should have the same window id included in their keys
        t_key=sorted(t_dic.keys() )[i] # extract the key of t_window
        f_key=sorted(f_dic.keys() )[i] # extract the key of f_window 
        
        t_window=t_dic[t_key] # extract the t_window
        f_window=f_dic[f_key] # extract the f_window

        window_user_id= int(t_key[-8:-6]) # extract the user id from window's key
        window_activity_id=int(t_key[-2:]) # extract the activity id from the windows key

        # generate all time features from t_window 
        time_features = t_axial_features_generation(t_window) + t_mag_features_generation(t_window)
        
        # generate all frequency features from f_window
        frequency_features = f_axial_features_generation(f_window) + f_mag_features_generation(f_window)
        
        # Generate addtional features from t_window
        additional_features = angle_features(t_window)
        # concatenate all features and append the activity id and the user id
        row= time_features + frequency_features + additional_features + [int(window_activity_id),int(window_user_id)]
        
        # go to the first free index in the dataframe
        free_index=len(final_Dataset)
        
        # append the row
        final_Dataset.loc[free_index]= row
    final_Dataset.drop(final_Dataset.iloc[:, 502:581], inplace = True, axis = 1) # Remove additional features
        
    return final_Dataset # return the final dataset