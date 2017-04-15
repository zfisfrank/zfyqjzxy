#usr/bin/python3
import numpy as np
import pandas as pd


def r_square(predictResult,target):
    targetMean = np.mean(target)
    return np.sum(np.square(predictResult-targetMean)) / np.sum(np.square(target-targetMean))

def reject_outliers(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def count_outliers(data, m=3):
    return np.sum(abs(data - np.mean(data)) > m * np.std(data))

# def count_outliers_df(self,df, m=3):
