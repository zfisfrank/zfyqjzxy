#/usr/bin/python3

from joblib import Parallel, delayed
from sklearn.linear_model import Ridge
from sklearn import ensemble
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn import svm
import mathFun as m
# declear models
def declareLO():
    ridGe = Ridge()
    svR = svm.SVR(C=5, gamma=0.001)
    adaBoost = ensemble.AdaBoostRegressor()
    bagging = ensemble.BaggingRegressor()
    extraTree = ensemble.ExtraTreesRegressor()
    gradientBoost = ensemble.GradientBoostingRegressor()
    randForest = ensemble.RandomForestRegressor()
    learningObjs = [svR,ridGe,adaBoost,bagging,extraTree,gradientBoost,randForest]
    return learningObjs

# learning function to store learnt result

def learnAlgos(trainFeature,trainTarget,valiFeature,valiTarget,testFeature,testTarget,targetText, learningObj):
    learningObj.fit(trainFeature,trainTarget[targetText])

    # trainPredictions
    trainPredictions = learningObj.predict(trainFeature)
    rSquareTrain = m.r_square(trainPredictions, trainTarget[targetText])
    meanSquareErrorTrain = m.mean_squared_error(trainTarget[targetText],trainPredictions)
    meanAbsErrorTrain = m.mean_absolute_error(trainTarget[targetText], trainPredictions)
    explainedVarScoreTrain = metrics.explained_variance_score(trainTarget[targetText], trainPredictions)
    medianAbsErrorTrain = m.median_absolute_error(trainTarget[targetText], trainPredictions)

    # validatePredictions
    valiPredictions = learningObj.predict(valiFeature)
    rSquareVali = m.r_square(valiPredictions, valiTarget[targetText])
    meanSquareErrorVali = m.mean_squared_error(valiTarget[targetText],valiPredictions)
    meanAbsErrorVali = m.mean_absolute_error(valiTarget[targetText], valiPredictions)
    explainedVarScoreVali = metrics.explained_variance_score(valiTarget[targetText], valiPredictions)
    medianAbsErrorVali = m.median_absolute_error(valiTarget[targetText], valiPredictions)

    # testPredictions
    testPredictions = learningObj.predict(testFeature)
    rSquareTest = m.r_square(testPredictions, testTarget[targetText])
    meanSquareErrorTest = m.mean_squared_error(testTarget[targetText],testPredictions)
    meanAbsErrorTest = m.mean_absolute_error(testTarget[targetText], testPredictions)
    explainedVarScoreTest = metrics.explained_variance_score(testTarget[targetText], testPredictions)
    medianAbsErrorTest = m.median_absolute_error(testTarget[targetText], testPredictions)

    outString = '\n' + learningObj.__class__.__name__  + ',' + \
                str(rSquareTrain) + ','+str(rSquareVali) + ','+str(rSquareTest) + ','+ \
                str(meanSquareErrorTrain) + ',' + str(meanSquareErrorVali) + ',' + str(meanSquareErrorTest) + ','+ \
                str(meanAbsErrorTrain)+ ',' + str(meanAbsErrorVali)+ ',' + str(meanAbsErrorTest)+ ','+ \
                str(explainedVarScoreTrain) + ',' + str(explainedVarScoreVali) + ',' + str(explainedVarScoreTest) + ',' + \
                str(medianAbsErrorTrain) + ',' + str(medianAbsErrorVali) + ',' + str(medianAbsErrorTest) + '\n'

    f= open("results/resultListGross.csv","a+")
    f.write(outString)
    f.close()
    print(outString)
    return


def testLagos(trainFeature,trainTarget,valiFeature,valiTarget,testFeature,testTarget,targetText):
    learningObjs = declareLO()
    f = open("results/resultListGross.csv", "a+")
    f.write('trainingAlgo,rSquareTrain,rSquareVali,rSquareTest,meanSquareErrorTrain,meanSquareErrorVali,meanSquareErrorTest,meanAbsErrorTrain,meanAbsErrorVali,meanAbsErrorTest,explainedVarScoreTrain,explainedVarScoreVali,explainedVarScoreTest,medianAbsErrorTrain,medianAbsErrorVali,medianAbsErrorTest')
    f.close()
    for learningObj in learningObjs:
        learnAlgos(trainFeature,trainTarget,valiFeature,valiTarget,testFeature,testTarget,targetText, learningObj)
    # Parallel(n_jobs=2)(delayed(learnAlgos)(trainFeature,trainTarget,valiFeature,valiTarget,testFeature,testTarget,targetText, learningObj) for learningObj in learningObjs)
