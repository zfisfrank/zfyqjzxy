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

tableCols = ('movieTittle','matchedTittle','MapNumber','Score')

simpleLookTable = pd.DataFrame(columns = tableCols)
tokenLookTable = pd.DataFrame(columns = tableCols)
simpleLookTable = pd.DataFrame()
tokenLookTable = pd.DataFrame()

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
    #if simpleR > 90:
    simpleLookTable = simpleLookTable.append(pd.Series([femaleName,maleNames[simpleI],simpleI,simpleR],index = tableCols),ignore_index=True)
    #if tokenR > 90:
    tokenLookTable = tokenLookTable.append(pd.Series([femaleName,maleNames[tokenI],tokenI,tokenR],index = tableCols),ignore_index=True)
    print('current is: '+ str(f) + 'th file')
score = tokenLookTable['Score']
tLT = tokenLookTable.loc[score>90,]
tLT.to_csv('90tresh.csv')
simpleLookTable.to_csv('simple.csv')
tokenLookTable.to_csv('token.csv')
