#usr/bin/python3
import numpy as np
import pandas as pd
#from sklearn import datasets, svm, metrics
from sklearn.cross_validation import train_test_split
# fullData = pd.read_csv('movie_metadata_clean_dup_gross_TV_missing_final.csv')
fullData = pd.read_excel('movie final_real final.xlsx')
# interestNumCols = ['num_critic_for_reviews', 'duration',
#        'director_facebook_likes', 'actor_3_facebook_likes',
#        'actor_1_facebook_likes', 'num_voted_users', 'cast_total_facebook_likes',
#        'facenumber_in_poster', 'num_user_for_reviews', 'title_year', 'actor_2_facebook_likes',
#        'aspect_ratio', 'movie_facebook_likes', 'budget_clean', ]

interestNumCols = ['num_critic_for_reviews', 'duration',
       'director_facebook_likes', 'actor_3_facebook_likes',
       'actor_1_facebook_likes', 'num_voted_users',
       'cast_total_facebook_likes', 'facenumber_in_poster',
       'num_user_for_reviews','title_year', 'actor_2_facebook_likes', 'aspect_ratio',
       'movie_facebook_likes', 'budget_clean2', 'movie_title', 'plot_keywords', 'movie_imdb_link', 'budget',
       'imdb_score', 'movie_title_clean', 'gross_clean_s12', 'log_gross_clean',
       'log_budget_clean', 'director_name','actor_2_name', 'actor_1_name','actor_3_name']

interestCatCols = ['language', 'country','content_rating','color']

catInfo = fullData[interestCatCols] # category Data initial
cateLabels = []
# convert 'language', 'country','content_rating'] to dummies
for idx in catInfo.columns:
    cateLabels.append(pd.get_dummies(catInfo[idx]))

# numlize genres also
genres = fullData['genres'].str.get_dummies()
cateLabels.append(genres)
# the cateLabels has been coverted to num for later use
cateLabels = pd.concat(cateLabels,axis=1)
# all the features
features = fullData[interestNumCols]
features = pd.concat([cateLabels,features],axis = 1)
trainData80, testData20 = train_test_split(features, test_size= .2)
trainData60, validateData20 = train_test_split(trainData80, test_size=0.25

trainData80.to_csv('./dataSet/trainData80.csv')
testData20.to_csv('./dataSet/testData20.csv')
trainData80.to_csv('./dataSet/trainData80.csv')
trainData80.to_csv('./dataSet/trainData80.csv')
