#/usr/bin/python3
import pandas as pd
import matplotlib.pyplot as plt


def readData():
    dropList = ['movie_title','plot_keywords','movie_imdb_link','director_name','actor_2_name','actor_1_name','actor_3_name','budget']
    trainData = pd.read_csv('dataSet/trainData60.csv').set_index('Unnamed: 0').set_index('movie_title_clean').drop(dropList,axis = 1)
    validateData = pd.read_csv('dataSet/validateData20.csv').set_index('Unnamed: 0').set_index('movie_title_clean').drop(dropList,axis = 1)
    testData = pd.read_csv('dataSet/testData20.csv').set_index('Unnamed: 0').set_index('movie_title_clean').drop(dropList,axis = 1)
    return [trainData, validateData, testData]

def splitFeatureTarget(data):
    feature = data.drop(['gross_clean_s12','log_gross_clean', 'imdb_score'], axis = 1)
    targets = data[['gross_clean_s12', 'log_gross_clean', 'imdb_score']]
    return [feature, targets]

def allDataSets():
    trainData,validateData, testData = readData()
    trainFeature,trainTarget = splitFeatureTarget(trainData)
    valiFeature,valiTarget = splitFeatureTarget(validateData)
    testFeature,testTarget = splitFeatureTarget(testData)
    return(trainFeature,trainTarget,valiFeature,valiTarget,testFeature,testTarget)


