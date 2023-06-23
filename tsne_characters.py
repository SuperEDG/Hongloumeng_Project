from itertools import chain

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

groups = {
    'g1': ['Xiren', 'Wangfuren', 'Qinkeqin'],
    'g2': ['Jiabaoyu', 'Liwan', 'Jiatanchun'],
    'g3': ['Lindaiyu', 'Qingwen', 'Xingfuren'],
    'g4': ['Miaoyu', 'Jiaxichun', 'Xuebaochai'],
    'g5': ['Youerjie', 'Shixiangyun', 'Jiayinchun'],
}


def plot_tsne_series(X, y, perplexies, n_iter=1000, alpha=0.1, palette=None):

    # we need to filter due to the t-SNE limitation.
    perplexies = list(filter(lambda item: item < len(X), perplexies))

    embs_X = []
    for p in perplexies:
        tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=p, n_iter=n_iter)
        emb_X = tsne.fit_transform(X)
        embs_X.append(emb_X)

    c1 = list(chain(*[list(embs_X[i][:, 0]) for i in range(len(perplexies))]))
    c2 = list(chain(*[list(embs_X[i][:, 1]) for i in range(len(perplexies))]))
    arr = list(chain(*[[p] * len(embs_X[0]) for p in perplexies]))

    tsne_data = pd.DataFrame()
    tsne_data["comp-1"] = c1
    tsne_data["comp-2"] = c2
    tsne_data["col"] = arr
    tsne_data["y"] = list(chain(*[y for p in perplexies]))

    g = sns.FacetGrid(tsne_data, col="col", hue="y", palette=palette)
    g.map(sns.scatterplot, "comp-1", "comp-2", alpha=alpha)
    g.add_legend()

    plt.show()


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def __find_group(name):
    assert(isinstance(name, str))
    for k, v in groups.items():
        if name in v:
            # group name.
            return ",".join(v)
    return "_Other"


df = pd.read_csv("model/LCPM/LCPM.csv")

X = []
y = []
for i, r in df.iterrows():
    X.append(normalize(np.array(r[1:]), 0, 1))
    y.append(r[0])

X = np.array(X)
plot_tsne_series(X=X, y=[__find_group(i) for i in y], perplexies=[2], n_iter=500, alpha=0.7,
                 palette={
                    "_Other": "gray",
                    ','.join(groups["g1"]): "red",
                    ','.join(groups["g2"]): "blue",
                    ','.join(groups["g3"]): "green",
                    ','.join(groups["g4"]): "orange",
                    ','.join(groups["g5"]): "purple",
                 })
