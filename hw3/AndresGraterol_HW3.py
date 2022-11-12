# ----------------------------
# Andres Graterol 
# CAP5610 - Fall 22
# 4031393
# ----------------------------
# Homework 3
# ----------------------------
import csv
import pandas as pd
import numpy as np

# QUESTION 1 
# Construct the k-means algorithm from scratch

# data.csv 
# 10,000 data samples (rows)
# 784 features per sample (columns)

# labels.csv 
# class label for all 10,000 points 
# labels are from 0-9

# Specify K = the number of categorical values of y 
# (The number of classifications; 10 labels)
# Data exists in 784 dimensions 
def retrieve_data_from_files(data_path, label_path):
    # Init lists that will hold our data 
    dataset = []
    labels = []

    # Populate the dataset
    with open(data_path, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            dataset.append(row)

    # Shape of data should be (10000,784)
    dataset = np.array(dataset)
    #print(dataset)
    print(dataset.shape)

    # Populate the labels 
    with open(label_path, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            labels.append(row)

    # Shape of labels should be (10000,)
    labels = np.array(labels)
    labels = labels.squeeze()
    #print(labels)
    print(labels.shape)

return dataset, labels 


def main():
    data_path = 'kmeans_data/data.csv'
    labels_path = 'kmeans_data/label.csv'

    dataset, labels = retrieve_data_from_files(data_path, labels_path)

    return None

main()