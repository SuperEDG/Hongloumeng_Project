from itertools import chain
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

# Define groups of characters
groups = {
    'g1': ['Jia Baoyu', 'Jia Yingchun', 'Jia Yuanchun', 'Jia Xichun', 'Xi Ren', 'Qin Keqing'],
    'g2': ['Lin Daiyu', 'Qing Wen', 'Miao Yu'],
    'g3': ['Xue Baochai', 'Li Wan', 'Jia She', 'Wang Furen', 'Xing Furen'],
    'g4': ['Shi Xiangyun', 'Xiang Ling', 'You Erjie', 'You Sanjie', 'Jia Tanchun', 'Jia Qiaojie'],
    'g5': ['Wang Xifeng', 'Jia Mu', 'Jia Zheng', 'Jia Zhen']
}

# Define function to plot t-SNE visualizations for different perplexity values
def plot_tsne_series(X, y, perplexies, n_iter=1000, alpha=0.1, palette=None):
    # Filter perplexities due to the t-SNE limitation.
    perplexies = list(filter(lambda item: item < len(X), perplexies))

    # Generate t-SNE embeddings for each perplexity
    embs_X = []
    for p in perplexies:
        tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=p, n_iter=n_iter, random_state=0)
        emb_X = tsne.fit_transform(X)
        embs_X.append(emb_X)

    # Create a flat list of x and y coordinates and corresponding perplexity values
    c1 = list(chain(*[list(embs_X[i][:, 0]) for i in range(len(perplexies))]))
    c2 = list(chain(*[list(embs_X[i][:, 1]) for i in range(len(perplexies))]))
    arr = list(chain(*[[p] * len(embs_X[0]) for p in perplexies]))

    # Prepare the DataFrame for plotting
    tsne_data = pd.DataFrame()
    tsne_data["comp-1"] = c1
    tsne_data["comp-2"] = c2
    tsne_data["perplexy"] = arr
    tsne_data["Groups"] = list(chain(*[y for p in perplexies]))

    # Create facet grid of scatter plots
    g = sns.FacetGrid(tsne_data, col="perplexy", hue="Groups", palette=palette)
    g.map(sns.scatterplot, "comp-1", "comp-2", alpha=alpha)

    # Move the legend to upper right
    plt.legend(loc='upper left')

    # Save the figure
    plt.gcf().set_size_inches(8, 6)
    plt.savefig("tsne.png", bbox_inches='tight', dpi=200)

# Function to normalize an array to a specified range
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

# Function to create a group name
def __make_group_name(v):
    return v[0] + " et. al."

# Function to determine which group a character belongs to
def __find_group(name):
    assert(isinstance(name, str))
    for k, v in groups.items():
        if name in v:
            # Return group name
            return __make_group_name(v)
    return "_Other"  # Default group name if character doesn't belong to any specified group

# Read in the data
df = pd.read_csv(r"C:\Users\Admin\Desktop\Project\LCPM-finall3.csv")

# Prepare X and y for the t-SNE plot
X = []
y = []
for i, r in df.iterrows():
    X.append(normalize(np.array(r[1:]), 0, 1))
    y.append(r[0])

# Convert to numpy array and plot
X = np.array(X)
plot_tsne_series(X=X, y=[__find_group(i) for i in y], perplexies=[2], n_iter=1000, alpha=0.7,
                 palette={
                    "_Other": "gray",
                    __make_group_name(groups["g1"]): "red",
                    __make_group_name(groups["g2"]): "blue",
                    __make_group_name(groups["g3"]): "green",
                    __make_group_name(groups["g4"]): "orange",
                    __make_group_name(groups["g5"]): "purple",
                 })
