#/usr/bin/python3
import numpy as np
import pandas as pd
#from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split

import string
from joblib import Parallel, delayed
from sklearn import preprocessing

# targetNums = list(range(1,27)) * 1000
# letter2NumMap = dict(zip(string.ascii_lowercase,targetNums))
# num2LetterMap = dict(zip(targetNums,string.ascii_lowercase))

fullData = pd.read_csv('movie_metadata_clean_dup_TV_gross_budget_final.csv')
interestNumCols = ['num_critic_for_reviews', 'duration',
       'director_facebook_likes', 'actor_3_facebook_likes',
       'actor_1_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes',
       'facenumber_in_poster', 'num_user_for_reviews', 'title_year', 'actor_2_facebook_likes',
       'aspect_ratio', 'movie_facebook_likes', 'budget_clean', ]

# convert cate informations, each category will have either 0 or 1
interestCatCols = ['language', 'country','content_rating']
catInfo = fullData[interestCatCols]
# le = []
cateLabels = []
# i = 0
for idx in catInfo.columns:
    cateLabels.append(pd.get_dummies(catInfo[idx]))
    # i = i+1
# the cateLabels has been coverted to num for later use
cateLabels = pd.concat(cateLabels,axis=1) #三个dataframe 合成一个
# all the features
features = pd.concat([cateLabels,fullData[interestNumCols]],axis = 1)
# features = fullData[interestNumCols]
# fill na with average values, if the value is na, then mean give the determination average weight
features = features.fillna(features.mean())
# just used as to save a copy of modified features
# features.to_csv('python_modified_features.csv')

movieScore = fullData['imdb_score'] #target 1
gross = fullData['gross_clean'] # target 2

trainData, testData, trainTarget, testTarget = train_test_split(features,movieScore,test_size= .5)
# trainDataGross, testDataGross, trainTargetGross, testTargetGross = train_test_split(features,gross,test_size= .5)

# claim learning objects here
# from sklearn.neural_network import MLPRegressor
# mlpR = MLPRegressor(hidden_layer_sizes = [800,800])
from sklearn import svm
svR = svm.SVR(C = 5,gamma = 0.001)
from sklearn.linear_model import Ridge
ridGe = Ridge()
from sklearn import ensemble
adaBoost = ensemble.AdaBoostRegressor()
bagging = ensemble.BaggingRegressor()
extraTree = ensemble.ExtraTreesRegressor()
gradientBoost = ensemble.GradientBoostingRegressor()
randForest = ensemble.RandomForestRegressor()

# group all learning objects in to list for for loop to use
# learningObjs = [mlpR,svR,ridGe,adaBoost,bagging,extraTree,gradientBoost,randForest]
learningObjs = [svR,ridGe,adaBoost,bagging,extraTree,gradientBoost,randForest]

# learning function to store learnt result
from sklearn import metrics
def learnALgos(learningObj,trainData, testData, trainTarget, testTarget):
    # print('start loop no. : ' + str(i))
    # gammaList = pd.read_csv('paraList.csv')
    #clf = svm.SVC(gamma = gammaList.iloc[i,1],C = gammaList.iloc[i,0],tol = 1e-5,coef0 = 0.1,kernel = 'sigmoid')
    #old one still work
    learningObj.fit(trainData,trainTarget)
    Predictions = learningObj.predict(testData)
    rSquare = metrics.r2_score(testTarget, Predictions)
    meanSquareError = metrics.mean_squared_error(testTarget, Predictions)
    outString = learningObj.__class__.__name__  + ',' +str(rSquare) + ',' + str(meanSquareError) + '\n'
    f= open("result_listScore.txt","a+")
    f.write(outString)
    f.close()
    print(outString)
#    return accuracy

# gammaList = pd.read_csv('paraList.csv')
Parallel(n_jobs=3)(delayed(learnALgos)(learnObj,trainData, testData, trainTarget, testTarget) for learnObj in learningObjs)

for learnObj in learningObjs:
    learnALgos(learnObj,trainData, testData, trainTarget, testTarget)

# for i in range(len(gammaList)):
#     svm_fit(i,trainData, testData, trainTarget, testTarget)

# acc = pd.Series(accuracies)
# acc.to_csv('results2.txt')
#print(layers)
# print(accuracies)
