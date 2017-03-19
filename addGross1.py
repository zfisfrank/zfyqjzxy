#! /bin/python3


from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
import numpy as np

X = pd.read_csv('movie_metadata.csv')
moreGross = pd.read_csv('MoreGross.csv')
femaleNames = X['movie_title']
femaleGross = X['gross']
nanIndex = np.isnan(femaleGross)
femaleNames = list(femaleNames[nanIndex]) # only select

maleNames = moreGross['Movie Title (click to view)']

tableCols = ('movieTittle','MapNumber','Score')

# simpleLookTable = pd.DataFrame(columns = tableCols)
# tokenLookTable = pd.DataFrame(columns = tableCols)
simpleLookTable = open('simple.csv','a')
tokenLookTable = open('token.csv','a')

for f in range(len(femaleNames)):
    femaleName = femaleNames[f]
    simpleR = 0
    tokenR = 0
    simpleI = 0
    tokenI = 0
    for i in range(len(maleNames)):
        maleName = maleNames[i]
        cSimpleR = fuzz.ratio(femaleName,maleName)
        cTokenR = fuzz.token_sort_ratio(femaleName,maleName)
        if (simpleR < cSimpleR):
            simpleR = cSimpleR
            simpleI = i
        if (tokenR < cTokenR):
            tokenR = cTokenR
            tokenI = i
    simpleStr =femaleName +','+ str(simpleI) + ',' + str(simpleR)+'\n'
    tokenStr = femaleName +','+str(tokenI) + ',' + str(simpleR)+'\n'
    simpleLookTable.write(simpleStr)
    tokenLookTable.write(tokenStr)
    print('current is: '+ str(f) + 'th file')
simpleLookTable.close()
tokenLookTable.close()
