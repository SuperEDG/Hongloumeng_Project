import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from copy import deepcopy
import random
import math

# Load data
data = pd.read_csv('./data/processed/component3-LCPM/LCPM.csv')
names = data['name']
traits = data.drop('name', axis=1)

# Calculate distance matrix
dist_matrix = pd.DataFrame(squareform(pdist(traits)), columns=names, index=names)

def calculate_total_distance(nodes):
    total_distance = 0
    for i in range(len(nodes)-1):
        total_distance += dist_matrix[nodes[i]][nodes[i+1]]
    return total_distance

def simulated_annealing(nodes, T=1000, cooling_rate=0.003, num_iter=10000):
    current_path = nodes
    current_distance = calculate_total_distance(nodes)
    
    for i in range(num_iter):
        # Cool down
        T *= 1-cooling_rate

        # Create new path
        new_path = deepcopy(current_path)
        a, b = np.random.randint(0, len(nodes), size=2)
        new_path[a], new_path[b] = new_path[b], new_path[a]
        new_distance = calculate_total_distance(new_path)

        # If the new path is better, accept it
        if new_distance > current_distance:
            current_path = new_path
            current_distance = new_distance
        # If the new path is worse, accept it with a probability
        else:
            if np.random.rand() < math.exp((new_distance - current_distance) / T):
                current_path = new_path
                current_distance = new_distance

        # If temperature is low, stop
        if T < 1:
            break

    return current_path, current_distance

# Number of characters to choose
n = 5

# Run simulated annealing
nodes = list(names[:n])
path, distance = simulated_annealing(nodes)
print("Selected characters:", path)