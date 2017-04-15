#usr/bin/python3
import numpy as np
import pandas as pd


def r_square(predictResult,target):
    targetMean = np.mean(target)
    return np.sum(np.square(predictResult-targetMean)) / np.sum(np.square(target-targetMean))

def reject_outliers(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def reject_training_set_outliers(feature, target, targetCol, m = 3):
    targetInterested = target[targetCol]
    targetInterested = targetInterested[pd.Series.abs(targetInterested - targetInterested.mean()) < m * targetInterested.std()]
    feature = feature.loc[targetInterested.index]
    target = target.loc[targetInterested.index]
    return feature,target




def count_outliers(data, m=3):
    return np.sum(abs(data - np.mean(data)) > m * np.std(data))

def mean_squared_error(predictResult,target):
    return np.mean(np.square(target - predictResult))

def mean_absolute_error(predictResult,target):
    return np.mean(np.abs(predictResult - target))

def median_absolute_error(predictResult,target):
    return np.median(np.abs(predictResult - target))

# def count_outliers_df(self,df, m=3):

