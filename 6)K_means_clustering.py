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

def allocate_centroids(fin_weights):
    centroids = []
    for i in range(k):
        centroids.append(list(fin_weights[random.randint(0, 5), random.randint(0, 5)]))
    return centroids


with open('myDataFrameForSOM.pkl', 'rb') as f:
    featureVector = pickle.load(f)

data = featureVector.values.tolist()

with open('SOM.pkl', 'rb') as f:
    som = pickle.load(f)
# plt.figure(figsize=(20, 20))
# plt.pcolor(som.distance_map().T, cmap='bone_r')

fin_weights = som.get_weights()


plt.show()
dbi = []
for k in range(8, 9):
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
        cluster_and_data = {}
        cluster_size = []
        for j in range(k):
            cluster_data = []

            for i in range(data.shape[0]):
                if allocate_cluster[i] == j:
                    cluster_data.append(data[i])
                    cluster_and_data[i] = j
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
    # dbi.append(dbs)
    print('Score for k = ' + str(k) + ' is ' + str(dbs))

keylist = sorted(cluster_and_data.keys())
final_list = []
for key in keylist:
    d = list(data[key])
    d.append(cluster_and_data[key])
    final_list.append(d)

with open('data_with_clusterid.pkl', 'wb') as f:
    pickle.dump(final_list, f)
    print('Savedf')
# final_k = dbi.index(min(dbi)) + 2
