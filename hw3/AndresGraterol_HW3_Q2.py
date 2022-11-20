# ----------------------------
# Andres Graterol 
# CAP5610 - Fall 22
# 4031393
# ----------------------------
# Homework 3
# QUESTION 2
# ----------------------------

import csv 
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

file_path = 'movies_data/ratings_small.csv'

# TODO: Check this separator 
#       Possibly use header = none 
#       Check stack overflow post 
#       Use double quotes?
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

data = Dataset.load_from_file(file_path, reader)

# Algorithm to use 
# PMF is simply the SVD with no bias
PMF = SVD(biased=False)

# By default, the RMSE and MAE measures are computed
# 5 K-Folds are used by default as well
cross_validate(algo=PMF, data=data, verbose=True)