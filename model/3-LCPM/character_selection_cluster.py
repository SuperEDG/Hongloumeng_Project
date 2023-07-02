import pandas as pd
from sklearn.cluster import KMeans

# 读取csv文件
df = pd.read_csv('./data/processed/component3-LCPM/LCPM.csv')


# 将角色名字设置为索引
df.set_index('Name', inplace=True)

# 确定要创建的聚类数
num_clusters = 5

# 运行k-means算法
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(df)

# 将聚类标签添加回原始数据集
df['cluster'] = kmeans.labels_

# 创建空的字典用于存储每个群组的角色名
clusters = {i: [] for i in range(num_clusters)}

# 遍历数据集，将每个角色的名字添加到对应的群组
for index, row in df.iterrows():
    clusters[row['cluster']].append(index)

# 打印每个群组的角色名
for cluster, names in clusters.items():
    print(f'Cluster {cluster}:')
    print(', '.join(names))
    print('\n')
