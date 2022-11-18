# ----------------------------
# Andres Graterol 
# CAP5610 - Fall 22
# 4031393
# ----------------------------
# Homework 3
# QUESTION 1 
# Construct the k-means algorithm from scratch

# data.csv 
# 10,000 data samples (rows)
# 784 features per sample (columns)

# labels.csv 
# class label for all 10,000 points 
# labels are from 0-9
# ----------------------------
import csv
import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
#import sklearn.metrics
import scipy.spatial as ss
#from scipy.special import logsumexp

def retrieve_data_from_files(data_path, label_path):
    # Init lists that will hold our data 
    dataset = []
    labels = []

    # Populate the dataset
    with open(data_path, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            int_row = [eval(i) for i in row]
            dataset.append(int_row)

    # Shape of data should be (10000,784)
    dataset = np.array(dataset)
    #print(dataset)
    #print("Shape of the dataset")
    print(dataset.shape)

    # Populate the labels 
    with open(label_path, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            int_row = [eval(i) for i in row]
            labels.append(int_row)

    # Shape of labels should be (10000,)
    #print("Shape of the labels")
    labels = np.array(labels)
    labels = labels.squeeze()
    #print(labels)
    print(labels.shape)

    return dataset, labels 

def euclidean_similarity(centers, point, similarity_type):
    similarity_array = []

    # Compute the similarity between the point and all the clusters
    for i in range(len(centers)):
        distance = ss.distance.cdist(np.expand_dims(centers[i], axis=0), np.expand_dims(point, axis=0), similarity_type)
        distance = distance.squeeze()
        similarity_array.append(distance)    

    similarity_array = np.array(similarity_array)
    similarity_array = similarity_array.squeeze()

    return similarity_array

def cosine_similarity(centers, dataset):
    return None

def jarcard_similarity(centers, dataset):
    return None

def compute_sse(centers, dataset, membership_matrix):
    sse = []

    # For each cluster
    for i in range(len(centers)):
        # Holds the sum for each cluster
        running_sum = [] 
        for j in range(len(dataset)):
            # If the point we are looking is a member of the current cluster
            if (membership_matrix[j][i] == 1):
                # Compute the distance between the point and cluster (answer will be 1 int)
                distance = ss.distance.cdist(np.expand_dims(centers[i], axis=0), np.expand_dims(dataset[j], axis=0), 'euclidean')
                distance = distance.squeeze()
                running_sum.append(distance)

        cluster_sum = sum(running_sum)
        sse.append(cluster_sum)

    # Sum the totals for each cluster
    sse_Total = sum(sse)
    sse_Total = np.array(sse_Total)
    print("Shape of sse total: ", sse_Total.shape)

    print("Final sse value:", sse_Total)
    return sse_Total

# This function computes new centroids based on given initial centers
def compute_new_centroids(centers, dataset, similarity_type):
    
    # Create 2D one-hot encoding array for cluster membership
    membership_matrix = [] 

    # For every point in the dataset 
    for point in range(len(dataset)):
        # Create array to store similarity to each centroid
        similarity_array = []

        # Create one-hot encoding cluster membership for each point 
        membership_array = np.zeros(10)

        # TODO: CONDENSE THESE AND JUST KEEP INVALID CHECK
        # Determine similarity to all the centers based on the metric  
        if (similarity_type == 'euclidean'):
            similarity_array = euclidean_similarity(centers, dataset[point], similarity_type)

        elif (similarity_type == 'cosine'):
            # TODO: Use the sklearn library!
            similarity_array = cosine_similarity()

        elif (similarity_type == 'jarcard'):
            similarity_array = jarcard_similarity()

        else:
            raise ValueError("Similarity measure is one that is not supported")

        # Determine the smallest number in the similarity array 
        # to determine cluster membership
        min_sim = similarity_array[0]
        index = 0
        cluster_member = 0
        for sim in range(len(similarity_array)):
            
            if (similarity_array[sim] < min_sim):
                min_sim = similarity_array[sim]
                cluster_member = index

            index += 1

        # Place a '1' to relating to the centroid we are assigning the point to
        membership_array[cluster_member] = 1 
        membership_matrix.append(membership_array)

    # Compute sse for these centers and all the points 
    sse = compute_sse(centers, dataset, membership_matrix)

    # Compute the new centers
    # Get the total count in each cluster 
    cluster_totals = [sum(x) for x in zip(*membership_matrix)]
    # Expect shape: (10,)
    cluster_totals = np.array(cluster_totals)

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

    #print("Sum per cluster shape:", sum_per_cluster.shape)
    new_centers = np.zeros((10, 784))

    # Divide each cluster row by the number of points in each cluster 
    for i in range(10):
        if (cluster_totals[i] != 0):
            new_centers[i] = [x / cluster_totals[i] for x in sum_per_cluster[i]] 

    return sse, new_centers

# This function computes the centroids and determines 
# when to stop the algorithm
def kmeans_algorithm(centers, dataset, similarity_measure):
    #new_centers = []
    #centers_history = []
    sse_list = []

    iterations = 0

    # TODO: Look at the kmeans.py example!
    while (1):
        print("Iteration:", iterations)
        sse, new_centers = compute_new_centroids(centers, dataset, similarity_measure)
        sse_list.append(sse)

        #centers_compared = centers.round(2)
        #new_centers_compared = centers.round(2)

        # TODO: Maybe round the centers??
        #if (new_centers.all() == centers.all()):
        #if(np.array_equal(new_centers, centers)):
        #if(np.allclose(new_centers_compared, centers_compared)):
        if(np.allclose(new_centers, centers)):
            print("Converged!")
            return centers, iterations, sse_list
        # Else, keep iterating
        else:
            #sse_list.append(sse)
            centers = new_centers
            iterations += 1

# Specify K = the number of categorical values of y 
# (The number of classifications; 10 labels)
# Data exists in 784 dimensions 
def main(similarity_measure):

    print("Let's begin")
    data_path = 'kmeans_data/data.csv'
    labels_path = 'kmeans_data/label.csv'

    dataset, labels = retrieve_data_from_files(data_path, labels_path)
    
    #dataset = dataset[:20]

    # Pick random centers from the dataset to start the algorithm 
    initial_centers = rnd.choices(dataset, k=10)
    initial_centers = np.array(initial_centers)

    print(initial_centers)
    # TODO: Keep track of iterations to answer question 3
    # TODO: Compute accuracies 
    centers, iterations, sse_list = kmeans_algorithm(initial_centers, dataset, similarity_measure)

    print("ALGORITHM FINISHED")
    print("Final Number of iterations:", iterations)
    print("Final Shape of centers: ", centers.shape)
    print("Length of sse_list:", len(sse_list))
    
    x = np.linspace(0, iterations+1, iterations+1)
    
    plt.scatter(x, sse_list)
    plt.show()

    return None

# The three different similarity measures that we will be using for this problem 
euclidean_similarity_string = 'euclidean'
cosine_similarity_string = 'cosine'
jarcard_similarity_string = 'jarcard'

main(euclidean_similarity_string)
#main(cosine_similarity_string)
#main(jarcard_similarity_string)