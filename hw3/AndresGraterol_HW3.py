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
import scipy.spatial as ss

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

def label_clusters(labels, membership_matrix):
    cluster_labels = []

    for i in range(10):
        labels_per_cluster = []
        for j in range(len(labels)):
            if (membership_matrix[j][i] == 1):
                labels_per_cluster.append(labels[j])

        # Check if this list is empty
        if not labels_per_cluster:
            # If so assign a label of 0 to the cluster
            labels_per_cluster.append(0)

        labels_per_cluster = np.array(labels_per_cluster)
        u, indices = np.unique(labels_per_cluster, return_inverse=True)
        cluster_labels.append(u[np.argmax(np.bincount(indices))])

    return cluster_labels

def compute_accuracy(dataset, labels, cluster_labels, membership_matrix):
    accuracy = 0

    for i in range(10):
        for j in range(len(dataset)):
            if (membership_matrix[j][i] == 1):
                if(cluster_labels[i] == labels[j]):
                    accuracy += 1

    # Number of correct predictions / number of items in dataset
    accuracy = accuracy / (len(dataset))

    return accuracy

# Compute similarity based on the type
def compute_similarity(centers, point, similarity_type):
    similarity_array = []

    # Compute the similarity between the point and all the clusters
    for i in range(len(centers)):
        distance = ss.distance.cdist(np.expand_dims(centers[i], axis=0), np.expand_dims(point, axis=0), similarity_type)
        distance = distance.squeeze()
        similarity_array.append(distance)    

    similarity_array = np.array(similarity_array)
    similarity_array = similarity_array.squeeze()

    return similarity_array

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

    print("SSE value:", sse_Total)
    return sse_Total

# This function computes new centroids based on given initial centers
def compute_new_centroids(centers, dataset, labels, similarity_type):
    
    # Create 2D one-hot encoding array for cluster membership
    membership_matrix = [] 

    # For every point in the dataset 
    for point in range(len(dataset)):
        # Create array to store similarity to each centroid
        similarity_array = []

        # Create one-hot encoding cluster membership for each point 
        membership_array = np.zeros(10)

        # Determine similarity to all the centers based on the metric  
        if (similarity_type != 'euclidean' and similarity_type != 'cosine' and similarity_type != 'jaccard'):
            raise ValueError("Similarity measure is one that is not supported")

        similarity_array = compute_similarity(centers, dataset[point], similarity_type)

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

    # Assign each cluster to a label
    cluster_labels = label_clusters(labels, membership_matrix)

    # Compute accuracies
    accuracy = compute_accuracy(dataset, labels, cluster_labels, membership_matrix)
    print("Accuracy: ", round(accuracy*100, 2))

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

    return sse, new_centers, accuracy

# This function computes the centroids and determines 
# when to stop the algorithm
def kmeans_algorithm(centers, dataset, labels, similarity_measure, stop_criteria):
    sse_list = []

    iterations = 0

    while (1):
        print("Iteration:", iterations)
        sse, new_centers, accuracy = compute_new_centroids(centers, dataset, labels, similarity_measure)
        #sse_list.append(sse)

        if (stop_criteria == 'centroid'):
            #centers_compared = centers.round(2)
            #new_centers_compared = centers.round(2)

            #if (new_centers.all() == centers.all()):
            #if(np.array_equal(new_centers, centers)):
            #if(np.allclose(new_centers_compared, centers_compared)):
            if(np.allclose(new_centers, centers)):
                print("Converged!")
                sse_list.append(sse)
                return iterations, sse_list, accuracy
            # Else, keep iterating
            else:
                #sse_list.append(sse)
                centers = new_centers
                iterations += 1
                sse_list.append(sse)
        
        elif (stop_criteria == 'sse'):
            if (sse > sse_list[-1]):
                print("SSE is now greater")
                sse_list.append(sse)
                return iterations, sse_list, accuracy
            else:
                centers = new_centers
                iterations += 1
                sse_list.append(sse)

        elif (stop_criteria == 'iteration'):
            if (iterations == 100):
                print("Reached max preset value")
                sse_list.append(sse)
                return iterations, sse_list, accuracy
            else:
                centers = new_centers
                iterations += 1
                sse_list.append(sse)
        
        else:
            raise ValueError("Invalid choice of stop criteria")

# Specify K = the number of categorical values of y 
# (The number of classifications; 10 labels)
# Data exists in 784 dimensions 
def main(similarity_measure, stop_criteria):

    print("Let's begin")
    data_path = 'kmeans_data/data.csv'
    labels_path = 'kmeans_data/label.csv'

    dataset, labels = retrieve_data_from_files(data_path, labels_path)

    # Pick random centers from the dataset to start the algorithm 
    initial_centers = rnd.choices(dataset, k=10)
    initial_centers = np.array(initial_centers)

    #print(initial_centers)
    iterations, sse_list, accuracy = kmeans_algorithm(initial_centers, dataset, labels, similarity_measure, stop_criteria)

    print("ALGORITHM FINISHED")
    print("Final Number of iterations:", iterations)
    print("Final Accuracy:", round(accuracy*100, 2))
    print("Initial SSE: ", sse_list[0])
    print("Final SSE: ", sse_list[-1])

    # Plot the SSE behavior
    x = np.linspace(0, iterations+1, iterations+1)
    plt.scatter(x, sse_list)
    plt.show()

    return None

# The three different similarity measures that we will be using for this problem 
euclidean_similarity_string = 'euclidean'
cosine_similarity_string = 'cosine'
jaccard_similarity_string = 'jaccard'

# The three different stopping criteria
centroid_stop_criteria = 'centroid'
sse_value_increase_criteria = 'sse'
preset_iteration_criteria = 'iteration'

main(jaccard_similarity_string, centroid_stop_criteria)
#main(cosine_similarity_string)
#main(jaccard_similarity_string)