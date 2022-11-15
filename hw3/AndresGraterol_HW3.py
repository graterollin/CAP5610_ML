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
#from scipy.special import logsumexp

# TODO: Take a look at below! ALSO LOOK FOR OTHER SCIKIT LEARN LIBRARIES THAT CAN HELP
#       FOR EXAMPLE, SIMILARITY HELP!!!
# Distance metric:
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html
# Cosine similarity: 
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
# TODO: LOOK AT KMEANS.PY
# TODO: ASK FOR AN EXTENSION ON THE HOMEWORK!!!!!!
# TODO: DECLUTTER THE CODE! REMOVE COMMENTS THAT WE KNOW FOR SURE ALREADY WORK!!!!!!!!!!!
# TODO: Possibly refactor the code for the membership matrix to get away from one-hot encoding mess
# TODO: SEE IF YOU CAN MANIPULATE THE DATA TO BE OF SHAPE (10000, 2 for example)

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
            #print("Feature #: ", j)
            distance = (centers[i][j] - point[j])**2
            distance_array.append(distance)
        
        sum_of_distances = sum(distance_array)
        similarity = np.sqrt(sum_of_distances)
        similarity_array.append(similarity)           

    similarity_array = np.array(similarity_array)
    # Shape goal: (10,)
    #print("Similarity array shape: ", similarity_array.shape)

    return similarity_array

def cosine_similarity(centers, dataset):
    return None

def jarcard_similarity(centers, dataset):
    return None

# Computes distance to only one center
def sse_distance_helper(center, point):
    distance_array = []

    # Loop through the features 
    for i in range(len(center)):
        distance = (center[i] - point[i])**2
        distance_array.append(distance)

    sum_of_distances = sum(distance_array)
    final_distance = np.sqrt(sum_of_distances)

    #print("Distance for point: ", final_distance)
    #print("Expecting 1 here: ")
    #print(final_distance.shape)

    return final_distance

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
                # TODO: Change this to sklearn distance metric
                distance = sse_distance_helper(centers[i], dataset[j])
                distance = distance.squeeze()
                running_sum.append(distance)

        cluster_sum = sum(running_sum)
        sse.append(cluster_sum)

    # Sum the totals for each cluster
    sse_Total = sum(sse)
    sse_Total = np.array(sse_Total)

    #print("Final sse value:", sse_Total)
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
        sse, new_centers = compute_new_centroids(centers, dataset, similarity_measure)
        sse_list.append(sse)

        print("Iteration:", iterations)
        #print("SSE List shape: ", len(sse_list))
        # return the final centers once they have converged (they do not change)
        #print("TIME TO COMPARE:")
        #print("New centers shape:", new_centers.shape)
        #print("Old centers shape: ", centers.shape)

        if (new_centers.all() == centers.all()):
            print("Converged!")
            return centers, iterations, sse_list
        # Else, keep iterating
        else:
            #sse_list.append(sse)
            centers = new_centers
            iterations += 1

    #return sse_list

# This function computes new centroids based on given initial centers
def compute_new_centroids(centers, dataset, similarity_type):
    
    # Create 2D one-hot encoding array for cluster membership
    membership_matrix = [] 

    # For every point in the dataset 
    for point in range(len(dataset)):
        # Create array to store similarity to each centroid
        #print("point # :", point)

        similarity_array = []

        # Create one-hot encoding cluster membership for each point 
        membership_array = np.zeros(10)

        # Determine similarity to all the centers based on the metric  
        if (similarity_type == 'Euclidean'):
            # TODO: Change this to sklearn distance metric 
            similarity_array = euclidean_similarity(centers, dataset[point])

        elif (similarity_type == 'Cosine'):
            # TODO: Use the sklearn library!
            similarity_array = cosine_similarity()

        elif (similarity_type == 'Jarcard'):
            similarity_array = jarcard_similarity()

        else:
            raise ValueError("Similarity measure is one that is not supported")

        # Determine the smallest number in the similarity array 
        # to determine cluster membership
        #print("Similarity array:", similarity_array)

        min_sim = similarity_array[0]
        index = 0
        for sim in range(len(similarity_array)):
            
            if (similarity_array[sim] < min_sim):
                min_sim = sim
                cluster_member = index

            index += 1

        # Place a '1' to relating to the centroid we are assigning the point to
        membership_array[cluster_member] = 1 
        membership_matrix.append(membership_array)

    #print("Final membership matrix:", membership_matrix)
    # TODO: Compute sse, create function
    # Compute sse for these centers and all the points 
    sse = compute_sse(centers, dataset, membership_matrix)

    # Compute the new centers
    # Get the total count in each cluster 
    cluster_totals = [sum(x) for x in zip(*membership_matrix)]
    cluster_totals = np.array(cluster_totals)

    # Expect shape: (10,)
    #print("Number per cluster shape:")
    #print(cluster_totals.shape)
    #print("Number per cluster:")
    #print(cluster_totals)

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

    #print("Sum per cluster shape:", sum_per_cluster.shape)
    new_centers = np.zeros((10, 784))

    # TODO: DEBUG HERE, THE NEW CENTERS ARE NOT UPDATING!!!!
    # Divide each cluster row by the number of points in each cluster 
    for i in range(10):
        if (cluster_totals[i] != 0):
            new_centers[i] = [x / cluster_totals[i] for x in sum_per_cluster[i]] 
    
    #print("Shape of new centers:")
    #print(new_centers.shape)

    return sse, new_centers

# Specify K = the number of categorical values of y 
# (The number of classifications; 10 labels)
# Data exists in 784 dimensions 
def main(similarity_measure):
    data_path = 'kmeans_data/data.csv'
    labels_path = 'kmeans_data/label.csv'

    dataset, labels = retrieve_data_from_files(data_path, labels_path)
    
    #dataset = dataset[:20]

    # Pick random centers from the dataset to start the algorithm 
    initial_centers = rnd.choices(dataset, k=10)
    initial_centers = np.array(initial_centers)
    #print("Shape of the initial centers:")
    #print(initial_centers.shape)

    # TODO: Keep track of iterations to answer question 3
    # TODO: Compute accuracies 
    centers, iterations, sse_list = kmeans_algorithm(initial_centers, dataset, similarity_measure)
    # Number of items in the sse_list will tell us how many times it ran before converging
    
    print("ALGORITHM FINISHED")
    print("Final Number of iterations:", iterations)
    print("Final Shape of centers: ", centers.shape)
    print("Length of sse_list:", len(sse_list))
    
    x = np.linspace(0, iterations+1, iterations+1)
    
    plt.scatter(x, sse_list)
    plt.show()

    return None

# The three different similarity measures that we will be using for this problem 
euclidean_similarity_string = 'Euclidean'
cosine_similarity_string = 'Cosine'
jarcard_similarity_string = 'Jarcard'

main(euclidean_similarity_string)
#main(cosine_similarity_string)
#main(jarcard_similarity_string)