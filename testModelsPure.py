# #/usr/bin/python3
import numpy as np
import pandas as pd
# #from sklearn import datasets, svm, metrics
# from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pylab
import scipy.stats as stats

from importlib import reload
import mathFun as m
import dataIO as io
import modelTest as mt
import numpy as np
from sklearn import ensemble
from sklearn import metrics
# init all objs used
def reInit():
    reload(m)
    reload(io)
    reload(mt)


# get all feature and targets
trainFeature,trainTarget,valiFeature,valiTarget,testFeature,testTarget = io.allDataSets()
# when modify module, need reload
reInit()
# testTargetName = 'gross_clean_s12'
# testTargetName = 'log_gross_clean'

testTargetName = 'imdb_score'
print(trainFeature.shape)
trainFeature,trainTarget = m.reject_training_set_outliers(trainFeature,trainTarget,testTargetName)
print(trainFeature.shape)
print(trainTarget.shape)

mt.testLagos(trainFeature,trainTarget,valiFeature,valiTarget,testFeature,testTarget,testTargetName)


# read different test values and combine
valueFileList = ['filterIMDBScore.csv','filterLogGross.csv','filterNormGross.csv','noFilterIMDBScore.csv','noFiltterLogGross.csv','noFiltterNormGross.csv']

# valueFileList = map(lambda x: 'results/experiments_procedure/'+x, valueFileList)
values = []
for f in valueFileList:
    df = pd.read_csv('results/experiments_procedure/'+ f)
    df['test'] = f.strip('.csv')
    # df = df.set_index(['test','trainingAlgo'])
    values.append(df)
    print(df.columns)
values2 = pd.concat(values,axis = 0, ignore_index=True)
values2.shape
values2  =values2.set_index(['test','trainingAlgo'])
values2.to_csv('results/experiments_procedure/all.csv')

# filtering
# normalize
from sklearn.preprocessing import normalize
a = normalize(trainFeature,norm = 'l1')
plt.plot(a)