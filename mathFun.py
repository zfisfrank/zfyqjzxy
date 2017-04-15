#usr/bin/python3
import numpy as np

def r_square(f,y):
    yMean = np.mean(y)
    return np.sum(np.square(f-yMean))/np.sum(np.square(y-yMean))