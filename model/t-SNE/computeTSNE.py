import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import combinations
from shapely.geometry import Polygon

# Function to calculate the area of a polygon given its vertices
def polygon_area(vertices):
    poly = Polygon(vertices)
    return poly.area

# Read data from CSV file
data = pd.read_csv('characters.csv')

# Extract the character names and features from the data
character_names = data.iloc[:, 0]
features = data.iloc[:, 1:]

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, perplexity=20)
reduced_features = tsne.fit_transform(features)

# Find the largest polygon formed by five characters
max_area = 0
selected_vertices = None

for comb in combinations(range(len(reduced_features)), 5):
    area = polygon_area(reduced_features[list(comb), :])
    if area > max_area:
        max_area = area
        selected_vertices = comb

# Visualize the t-SNE results
plt.figure(figsize=(10, 6))
plt.scatter(reduced_features[:, 0], reduced_features[:, 1])

# Add character names as labels to the plot
for i, name in enumerate(character_names):
    plt.annotate(name, (reduced_features[i, 0], reduced_features[i, 1]))

# Connect the five characters with lines and fill the area
for i in range(5):
    plt.plot(*zip(reduced_features[selected_vertices[i]], reduced_features[selected_vertices[(i+1) % 5]]), 'k-')
plt.fill(reduced_features[list(selected_vertices), 0], reduced_features[list(selected_vertices), 1], 'c', alpha=0.3)

# Set plot title and axis labels
plt.title('t-SNE Visualization of Character Features with Largest Polygon')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# Display the plot
plt.show()

# Print the selected character names
print("Selected characters:")
for i in selected_vertices:
    print(character_names[i])
