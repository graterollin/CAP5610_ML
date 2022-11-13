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
import random as rnd

# QUESTION 1 
# Construct the k-means algorithm from scratch

# data.csv 
# 10,000 data samples (rows)
# 784 features per sample (columns)

# labels.csv 
# class label for all 10,000 points 
# labels are from 0-9

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

def euclidean_similarity():
    return None

def cosine_similarity():
    return None

def jarcard_similarity():
    return None

def compute_SSE(centers, dataset):
    return None

# This function computes the centroids and determines 
# when to stop the algorithm
# TODO: Keep track of SSE listing here as well
def kmeans_algorithm(centers, dataset, similarity_measure):
    #new_centers = []
    #centers_history = []
    SSE_list = []

    iterations = 0

    while (1):
        new_centers, SSE = compute_new_centroids(centers, dataset, similarity_measure)

        # return the final centers once they have converged (they do not change)
        if (new_centers == centers):
            return centers, iterations, SSE_list
        # Else, keep iterating
        else:
            SSE_list.append(SSE)
            centers = new_centers
            iterations += 1

        # TODO: What if we return an empty list?
        # return the final centers once they have converged
        #if (centers_history.pop() == centers):
        #    return centers

        #centers_history.append(centers)

    return None

# This function computes new centroids based on given initial centers
def compute_new_centroids(centers, dataset, similarity_measure):
    
    # Create 2D one-hot encoding array for cluster membership
    membership_matrix = [] 

    # For every point in the dataset 
    for point in range(len(dataset)):
        # Create array to store similarity to each centroid
        similarity_array = []

        # Create one-hot encoding cluster membership for each point 
        membership_array = np.zeros(10)

        # Determine similarity to all the centers based on the metric  
        if (similarity_measure == 'Euclidean'):
            similarity_array = euclidean_similarity()

        elif (similarity_measure == 'Cosine'):
            similarity_array = cosine_similarity()

        elif (similarity_measure == 'Jarcard'):
            similarity_array = jarcard_similarity()

        else:
            raise ValueError("Similarity measure is one that is not supported")

        # Determine the smallest number in the similarity array 
        # to determine cluster membership
        min_sim = similarity_array[0]
        index = 0
        for sim in range(len(similarity_array)):
            
            if (sim < min_sim):
                min_sim = sim
                cluster_member = index

            index += 1

        # Place a '1' to relating to the centroid we are assigning the point to
        membership_array[cluster_member] = 1 
        membership_matrix.append(membership_array)

    # TODO: Compute SSE, create function
    # Compute SSE for these centers and all the points 
    SSE = compute_SSE(centers, dataset, membership_matrix)

    # Compute the new centers
    # Get the total count in each cluster 
    cluster_totals = [sum(x) for x in zip(*membership_matrix)]
    # Expect shape: (10,)
    print("Number per cluster shape:")
    print(cluster_totals.shape)
    print("Number per cluster:")
    print(cluster_totals)

    # 10 rows (one for each cluster)
    # 784 columns 
    # Shape: (10, 784)
    # TODO: Check if this is initiated right 
    sum_per_cluster = np.zeros((10, 784))

    # Get the sum for each cluster
    # One column of the membership matrix at a time
    # Fills the sums per cluster row at a time
    for i in range(10):
        # represents the rows of the membership matrix 
        for j in range(len(dataset)):
            if (membership_matrix[j][i] == 1):
                # Add to the cluster row    The point in row j  
                sum_per_cluster[i] = np.add(sum_per_cluster[i], dataset[j])

    new_centers = np.zeros((10, 784))

    # Divide each cluster row by the number of points in each cluster 
    for i in range(10):
        new_centers[i] = [x / cluster_totals[i] for x in sum_per_cluster[i]] 
    
    print("Shape of new centers:")
    print(new_centers.shape)

    return SSE, new_centers

# Specify K = the number of categorical values of y 
# (The number of classifications; 10 labels)
# Data exists in 784 dimensions 
def main(similarity_measure):
    data_path = 'kmeans_data/data.csv'
    labels_path = 'kmeans_data/label.csv'

    dataset, labels = retrieve_data_from_files(data_path, labels_path)
    
    # Pick random centers from the dataset to start the algorithm 
    initial_centers = rnd.choices(dataset, k=10)
    initial_centers = np.array(initial_centers)
    print("Shape of the initial centers:")
    print(initial_centers.shape)

    # TODO: Keep track of iterations to answer question 3
    #SSE_list = 

    return None

# The three different similarity measures that we will be using for this problem 
euclidean_similarity = 'Euclidean'
cosine_similarity = 'Cosine'
jarcard_similarity = 'Jarcard'

main(euclidean_similarity)
#main(cosine_similarity)
#main(jarcard_similarity)