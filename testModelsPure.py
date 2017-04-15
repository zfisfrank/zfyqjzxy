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
import numpy as np
from sklearn import ensemble
from sklearn import metrics
# init all objs used
def reInit():
    reload(m)
    reload(io)

# when modify module, need reload
reInit()

# get all feature and targets
trainFeature,trainTarget,valiFeature,valiTarget,testFeature,testTarget = io.allDataSets()

# all targets amount of outlier
# m.count_outliers(trainTarget)

# declare a learning object
gradientBoost = ensemble.GradientBoostingRegressor()
# test two targets differences
def testTargets(trainFeature,trainTarget,valiFeature,valiTarget,testFeature,testTarget,targetText, learningObj):
    learningObj.fit(trainFeature,trainTarget[targetText])
    prediction = learningObj.predict(valiFeature)

    myRsq = m.r_square(prediction,valiTarget[targetText])
    avaRsq = metrics.r2_score(prediction,valiTarget[targetText])

    print(myRsq,avaRsq)
    return myRsq,avaRsq

print(testTargets(trainFeature,trainTarget,valiFeature,valiTarget,testFeature,testTarget, 'gross_clean_s12', gradientBoost))