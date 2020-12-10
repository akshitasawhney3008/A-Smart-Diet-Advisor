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
number_of_features = len(data[0])
train_data_len = len(data)
number_of_neurons = int(len(data)**(1/5))

som = MiniSom(number_of_neurons, number_of_neurons, len(data[0]), sigma=0.5, learning_rate=0.001)
print('Training')
i = 100
qe = []
iterations = []
while(i<=1000000):
    print(i)
    som.train_batch(data, i)
    iterations.append(i)
    qe.append(som.quantization_error(np.asarray(data)))
    i = i*10

plt.plot(iterations, qe, 'red')
plt.xlabel('iterations')
plt.ylabel('quantization error')
plt.title('To select the right number of iterations')
plt.legend()
plt.show()
