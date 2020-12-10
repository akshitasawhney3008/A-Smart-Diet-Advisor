from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import davies_bouldin_score
from scipy.spatial import distance
import pandas as pd
from scipy.spatial.distance import euclidean


def DaviesBouldin(X, labels, nc):
    n_cluster = nc
    cluster_k = [X[labels == k] for k in range(n_cluster)]
    centroids_np = [np.mean(k, axis=0) for k in cluster_k]
    variances = [np.mean([euclidean(p, centroids_np[i]) for p in k]) for i, k in enumerate(cluster_k)]
    db = []
    for i in range(n_cluster):
        for j in range(n_cluster):
            if j != i:
                db.append((variances[i] + variances[j]) / euclidean(centroids_np[i], centroids_np[j]))
    return np.max(db) / n_cluster


# fileName = 'final_normalized_nut_data.tsv'
# myDataFrame = pd.read_csv(fileName, sep='\t')
# myDataFrame.drop('Serving Size', axis=1, inplace=True)
# myDataFrame.drop(myDataFrame.columns[len(myDataFrame.columns)-1], axis=1, inplace=True)
# maxEnergy = myDataFrame['Energy'].max()
# for i in range(myDataFrame.shape[0]):
#     myDataFrame.iat[i, 15] = myDataFrame.iat[i, 15]/maxEnergy
# with open('myDataFrame.pkl', 'wb') as f:
#     pickle.dump(myDataFrame, f)
# print('Done')

# with open('myDataFrame.pkl', 'rb') as f:
#     myDataFrame = pickle.load(f)
#
# featureVector = myDataFrame.iloc[:, 2:16].copy()
# featureVector = featureVector[['Energy', 'Protein', 'Total lipid (fat)', 'Carbohydrate, by difference',
#                                'Fiber, total dietary', 'Vitamin C, total ascorbic acid']]
#
# with open('myDataFrameForSOM.pkl', 'wb') as f:
#     pickle.dump(featureVector, f)

with open('myDataFrameForSOM.pkl', 'rb') as f:
    featureVector = pickle.load(f)

data = featureVector.values.tolist()
print(data[1])
number_of_features = len(data[0])
train_data_len = len(data)
number_of_neurons = int(len(data)**(1/5))

som = MiniSom(number_of_neurons, number_of_neurons, len(data[0]), sigma=0.5, learning_rate=0.001)
print('Training')
som.train_batch(data, 10000)
fin_weights = som.get_weights()
print('Ready and Saving')

with open('SOM.pkl', 'wb') as f:
    pickle.dump(som, f)
    print('Saved')

with open('SOM.pkl', 'rb') as f:
    som = pickle.load(f)
plt.figure(figsize=(20, 20))
plt.pcolor(som.distance_map().T, cmap='bone_r')
quantization_err = som.quantization_error(np.asarray(data))
print(quantization_err)

def allocate_centroids(fin_weights):
    centroids = []
    for i in range(k):
        centroids.append(list(fin_weights[random.randint(0, 5), random.randint(0, 5)]))
    return centroids

plt.show()
dbi = []
for k in range(8, 19):
    print("k= ", k)
    centroids = allocate_centroids(fin_weights)
    diff = np.ones((k, 1))
    delta = (diff + 0.001)-1
    count = 0
    while(diff > delta).all():
        flag = 0
        distance_matrix = []
        allocate_cluster = []
        count = count + 1
        print("count = ",count)
        for centre in centroids:
            centre = np.asarray(centre).reshape(1, -1)
            data = np.asarray(data)
            distances = euclidean_distances(data, centre)
            distances = sum(distances.tolist(), [])
            distance_matrix.append(distances)
        distance_matrix = np.asarray(distance_matrix).transpose()
        for i in range(distance_matrix.shape[0]):
            min_cl = np.min(distance_matrix[i], axis=0)
            dis = list(distance_matrix[i])
            allocate_cluster.append(dis.index(min_cl))
        allocate_cluster = np.asarray(allocate_cluster)

        new_centroids = []
        k_clusters = []
        cluster_size = []
        for j in range(k):
            cluster_data = []
            for i in range(data.shape[0]):
                if allocate_cluster[i] == j:
                    cluster_data.append(data[i])
            if len(cluster_data) == 0 :
                centroids = allocate_centroids(fin_weights)
                flag = 1
                break
            cluster_size.append(len(cluster_data))
            k_clusters.append(cluster_data)
            cluster_data = np.asarray(cluster_data)
            new_centroids.append(np.mean(cluster_data, axis=0))
        print("size_of_clusters: ", cluster_size)
        for i in range(k):
            if flag == 1:
                break
            c1 = np.asarray(new_centroids)[i]
            c2 = np.asarray(centroids)
            c2 = c2[i]
            diff[i] = euclidean_distances(c1.reshape(1,-1), c2.reshape(1,-1))
        if (diff > delta).all():
            if flag != 1:
                centroids = new_centroids
    dbs = DaviesBouldin(data, allocate_cluster, k)
    dbi.append(dbs)
    print('Score for k = ' + str(k) + ' is ' + str(dbs))
final_k = dbi.index(min(dbi)) + 2
