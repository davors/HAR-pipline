import numpy as np

def readSample(line):
    row=[]
    row.append([float(element) for element in line.split()])
    #data=np.array(row)


    # Create a pandas dataframe from this 2D numpy array with column names
    #data_frame=pd.DataFrame(data=data,columns=columns)

    # return the data frame
    return row