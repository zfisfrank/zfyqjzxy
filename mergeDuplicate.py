#! /bin/python3
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd
import numpy as np

X = pd.read_csv('movie_metadata.csv')
a = X.drop_duplicates(subset = 'movie_title', keep = 'first')
a.to_csv('uniqueMetadata.csv')
