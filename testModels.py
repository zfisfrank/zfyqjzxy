#/usr/bin/python3
import numpy as np
import pandas as pd
#from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
from sklearn import svm
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
cateLabels = pd.concat(cateLabels,axis=1)
# all the features
features = pd.concat([cateLabels,fullData[interestNumCols]],axis = 1)
# fill na with average values, if the value is na, then mean give the determination average weight
features = features.fillna(features.mean())
features.to_csv('python_modified_features.csv')
target1 = fullData['imdb_score']
target2 = fullData['gross_clean']

# target3 = ['logGross']




dataId = fullData['Id']
target = fullData['Prediction']
# target = target.map(letter2NumMap)
data = fullData.drop(['Id', 'Prediction', 'NextId', 'Position'],axis = 1)
#data = fullData.drop(['Id', 'Prediction'],axis = 1)

"""this part to test current algorithms' accuracies"""
accuracies = []


trainData, testData, trainTarget, testTarget = train_test_split(data,target,test_size= .5)

# clf = svm.SVC(C=1,gamma=0.01)
def svm_fit(i,trainData, testData, trainTarget, testTarget):
    print('start loop no. : ' + str(i))
    gammaList = pd.read_csv('paraList.csv')
    #clf = svm.SVC(gamma = gammaList.iloc[i,1],C = gammaList.iloc[i,0],tol = 1e-5,coef0 = 0.1,kernel = 'sigmoid')
    #old one still work
    clf = svm.SVC(gamma = gammaList.iloc[i,1],C = gammaList.iloc[i,0],tol = 1e-2)
    clf.fit(trainData,trainTarget)
    Predictions = clf.predict(testData)
    accuracy = sum(Predictions == testTarget)/len(testTarget)
    outString = str(gammaList.iloc[i,1]) + ',' + str(gammaList.iloc[i,0]) + ',' +str(accuracy) + '\n'
    f= open("smvParaRunResult.txt","a+")
    f.write(outString)
    f.close()
    print(outString)
#    return accuracy

gammaList = pd.read_csv('paraList.csv')
Parallel(n_jobs=12)(delayed(svm_fit)(i,trainData, testData, trainTarget, testTarget) for i in range(len(gammaList)))
# for i in range(len(gammaList)):
#     svm_fit(i,trainData, testData, trainTarget, testTarget)

# acc = pd.Series(accuracies)
# acc.to_csv('results2.txt')
#print(layers)
# print(accuracies)
