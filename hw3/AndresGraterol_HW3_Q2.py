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