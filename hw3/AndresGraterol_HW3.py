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
            int_row = [eval(i) for i in row]
            dataset.append(int_row)

    # Shape of data should be (10000,784)
    dataset = np.array(dataset)
    #print(dataset)
    print("Shape of the dataset")
    print(dataset.shape)

    # Populate the labels 
    with open(label_path, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            int_row = [eval(i) for i in row]
            labels.append(int_row)

    # Shape of labels should be (10000,)
    print("Shape of the labels")
    labels = np.array(labels)
    labels = labels.squeeze()
    #print(labels)
    print(labels.shape)

    return dataset, labels 

def euclidean_similarity(centers, point):
    similarity_array = []

    # Compute the similarity between the point and all the clusters
    #print("Want this to be 10: ")
    #print(len(centers))
    #print("Want this to be 784:")
    #print(len(centers[0]))
    for i in range(len(centers)):
        # compare every feature for every cluster
        distance_array = []
        for j in range(len(centers[i])):
            distance = (centers[i][j] - point[j])**2
            distance_array.append(distance)
        
        sum_of_distances = sum(distance_array)
        similarity = np.sqrt(sum_of_distances)
        similarity_array.append(similarity)           

    similarity_array = np.array(similarity_array)
    # Shape goal: (10,)
    # unless being called by sse
    #print("Similarity array shape: ", similarity_array.shape)

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
                distance = euclidean_similarity(centers[i], dataset[j])
                distance = distance.squeeze()
                running_sum.append(distance)

        cluster_sum = sum(running_sum)
        sse.append(cluster_sum)

    # Sum the totals for each cluster
    sse_Total = sum(sse)
    sse_Total = np.array(sse_Total)

    print("Final sse value:", sse_Total.shape)
    return sse_Total

# This function computes the centroids and determines 
# when to stop the algorithm
# TODO: Keep track of sse listing here as well
def kmeans_algorithm(centers, dataset, similarity_measure):
    #new_centers = []
    #centers_history = []
    sse_list = []

    iterations = 0

    while (1):
        new_centers, sse = compute_new_centroids(centers, dataset, similarity_measure)
        sse_list.append(sse)

        print("Iteration:", iterations)

        # return the final centers once they have converged (they do not change)
        if (new_centers == centers):
            return centers, iterations, sse_list
        # Else, keep iterating
        else:
            #sse_list.append(sse)
            centers = new_centers
            iterations += 1

        # TODO: What if we return an empty list?
        # return the final centers once they have converged
        #if (centers_history.pop() == centers):
        #    return centers

        #centers_history.append(centers)

    return sse_list

# This function computes new centroids based on given initial centers
def compute_new_centroids(centers, dataset, similarity_type):
    
    # Create 2D one-hot encoding array for cluster membership
    membership_matrix = [] 

    # For every point in the dataset 
    for point in range(len(dataset)):
        # Create array to store similarity to each centroid
        print("point# :", point)

        similarity_array = []

        # Create one-hot encoding cluster membership for each point 
        membership_array = np.zeros(10)

        # Determine similarity to all the centers based on the metric  
        if (similarity_type == 'Euclidean'):
            similarity_array = euclidean_similarity(centers, dataset[point])

        elif (similarity_type == 1):
            similarity_array = cosine_similarity()

        elif (similarity_type == 2):
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

    # TODO: Compute sse, create function
    # Compute sse for these centers and all the points 
    sse = compute_sse(centers, dataset, membership_matrix)

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

    return sse, new_centers

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
    # TODO: Compute accuracies 
    sse_list = kmeans_algorithm(initial_centers, dataset, similarity_measure)
    # Number of items in the sse_list will tell us how many times it ran before converging
    number_of_iterations = len(sse_list)

    # TODO: Plot sse per iteration once code works

    return None

# The three different similarity measures that we will be using for this problem 
euclidean_similarity_string = 'Euclidean'
cosine_similarity_string = 'Cosine'
jarcard_similarity_string = 'Jarcard'

main(euclidean_similarity_string)
#main(cosine_similarity_string)
#main(jarcard_similarity_string)