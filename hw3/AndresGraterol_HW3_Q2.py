# ----------------------------
# Andres Graterol 
# CAP5610 - Fall 22
# 4031393
# ----------------------------
# Homework 3
# QUESTION 2
# ----------------------------
import numpy as np
import matplotlib.pyplot as plt
from surprise import SVD, Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate

file_path = 'movies_data/ratings_small.csv'

reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

data = Dataset.load_from_file(file_path, reader)

# Probabilistic Matrix Factorization
# -----------------------------------------------------
# Algorithm to use 
# PMF is simply the SVD with no bias
PMF = SVD(biased=False)

# By default, the RMSE and MAE measures are computed
# 5 K-Folds are used by default as well
#cross_validate(algo=PMF, data=data, verbose=True)

# User-Based Collaborative Filtering 
# -----------------------------------------------------
user_based = {
    "name": "cosine",
    "user_based": True,
}

# By default 40 neighbors are used
userBased_filtering = KNNBasic(k=60, sim_options=user_based)

#cross_validate(algo=userBased_filtering, data=data, verbose=True)

# Item-Based Collaborative Filtering 
# -----------------------------------------------------
item_based = {
    "name": "msd",
    "user_based": False,
}

itemBased_filtering = KNNBasic(k=60, sim_options=item_based)

#cross_validate(algo=itemBased_filtering, data=data, verbose=True)

# Plot the data for similarity/neighbor comparison 
# -----------------------------------------------------
sim_values = ['MSD', 'Cosine', 'Pearson']
user_rmse_data = [1.0078, 0.9935, 0.9983]
user_mae_data = [0.7784, 0.7675, 0.7727]

X_axis = np.arange(len(sim_values))

plt.bar(X_axis - 0.2, user_rmse_data, 0.4, label = 'RMSE')
plt.bar(X_axis + 0.2, user_mae_data, 0.4, label = 'MAE')

plt.xticks(X_axis, sim_values)
plt.xlabel("Sim Measures")
plt.ylabel("Values")
plt.title("User Based Collaborative Filtering")
plt.legend()
plt.show()

item_rmse_data = [0.9340, 0.9947, 0.9890] 
item_mae_data =[0.7206, 0.7751, 0.7678]

plt.bar(X_axis - 0.2, item_rmse_data, 0.4, label = 'RMSE')
plt.bar(X_axis + 0.2, item_mae_data, 0.4, label = 'MAE')

plt.xticks(X_axis, sim_values)
plt.xlabel("Sim Measures")
plt.ylabel("Values")
plt.title("Item Based Collaborative Filtering")
plt.legend()
plt.show()

neighbor_numbers = ['20', '40', '60']
userNeighbor_rmse_data = [0.9980, 0.9935, 0.9950]
userNeighbor_mae_data = [0.7698, 0.7675, 0.7696]

X_axis = np.arange(len(neighbor_numbers))

plt.bar(X_axis - 0.2, userNeighbor_rmse_data, 0.4, label = 'RMSE')
plt.bar(X_axis + 0.2, userNeighbor_mae_data, 0.4, label = 'MAE')

plt.xticks(X_axis, neighbor_numbers)
plt.xlabel("Neighbor Numbers")
plt.ylabel("Values")
plt.title("User Based Collaborative Filtering")
plt.legend()
plt.show()

itemNeighbor_rmse_data = [0.9479, 0.9340, 0.9332]
itemNeighbor_mae_data = [0.7324, 0.7206, 0.7192]

plt.bar(X_axis - 0.2, itemNeighbor_rmse_data, 0.4, label = 'RMSE')
plt.bar(X_axis + 0.2, itemNeighbor_mae_data, 0.4, label = 'MAE')

plt.xticks(X_axis, neighbor_numbers)
plt.xlabel("Neighbor Numbers")
plt.ylabel("Values")
plt.title("Item Based Collaborative Filtering")
plt.legend()
plt.show()