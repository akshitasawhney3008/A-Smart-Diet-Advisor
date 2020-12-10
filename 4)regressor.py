import numpy as np
import pandas as pd
import math
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA
# import seaborn as sns

my_file = np.genfromtxt('nutrients_train_data.tsv',dtype='unicode',delimiter='\t')
my_file = my_file.astype(np.float)
np.random.shuffle(my_file)

# with open('myDataFrameForSOM.pkl', 'rb') as f:
#     featureVector = pickle.load(f)
# my_file = featureVector.values
# my_file = my_file.astype(np.float)
# energy = my_file[:,0]
# my_file = my_file[:,1:]
# energy = my_file1[:,-1]
# energy = energy.reshape((energy.shape[0],1))
# my_file = np.append(my_file,energy,axis=1)
# np.random.shuffle(my_file)
# my_file = pd.DataFrame(my_file)
# sns.pairplot(my_file)
# X = my_file
feature_list = [0,1,2,4,9]
X = my_file[:,:-1]
X = X[:, feature_list]
Y = my_file[:,-1]
# Y = energy
Y = Y.astype(np.float)


# pca1 = PCA(n_components=1)
# principalComponents = pca1.fit_transform(X)

# # principalDf = pd.DataFrame(data=principalComponents
# #              , columns=['principal component 1'])
# # principalDf = principalDf.as_matrix()
# # target = pd.DataFrame(data=Y, columns=['target'])
# target = Y
# # finalDf = pd.concat([principalDf, target], axis=1)
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 Component PCA', fontsize = 20)
# ax.scatter(principalComponents, target, s=50,c='r')



# #tsne
# tsne = TSNE(n_components=1, random_state=0)
# X_2d = tsne.fit_transform(X)
# # tsneDf = pd.DataFrame(data=X_2d
# #              , columns=['tsne component 1', 'tsne component 2'])
# # target = pd.DataFrame(data=Y, columns=['target'])
# # finalDf = pd.concat([tsneDf, target], axis=1)
# target = Y
# fig1 = plt.figure(figsize = (8,8))
# ax = fig1.add_subplot(1,1,1)
# ax.set_xlabel('TSNE Component 1', fontsize=15)
# ax.set_ylabel('Energy', fontsize=15)#ye theek krdo fir
# ax.set_title('2 Component TSNE', fontsize=20)
# ax.scatter(X_2d, target, s=50,c='r')
# # targets = [0, 1]
# # targetnames = ["Early Stage","Late stage"]
# # colors = ['r', 'g', 'b']
# # for target, color in zip(targets,colors):
# #     indicesToKeep = finalDf['target'] == target
# #     ax.scatter(finalDf.loc[indicesToKeep, 'tsne component 1']
# #                , finalDf.loc[indicesToKeep, 'tsne component 2']
# #                , c=color
# #                , s=50)
# #     ax.legend(targetnames)
# #     ax.grid()
# plt.show()
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X,Y,test_size=0.70)


train_rmse = []
test_rmse = []
holdout_rmse = []
degree_list = []
for deg in range(1, 2):
    kf = KFold(n_splits=5)
    sum = 0
    rmse_train_list = []
    rmse_test_list = []
    Y_train1 = np.reshape(Y_train1,(Y_train1.shape[0], 1))
    Y_test1 = np.reshape(Y_test1,(Y_test1.shape[0], 1))
    lm = LinearRegression()
    poly = PolynomialFeatures(degree=deg)
    degree_list.append(deg)
    train_data = np.append(X_train1,Y_train1,axis=1)
    for train,test in kf.split(train_data):
        X_train = train_data[train][:,:-1]
        X_test = train_data[test][:,:-1]
        Y_train = train_data[train][:,-1]
        Y_test = train_data[test][:,-1]
        X_ = poly.fit_transform(X_train)
        X_test_ = poly.fit_transform(X_test)
        # lm = lm.fit(X_,Y_train)
        lm = lm.fit(X_, Y_train)
        predictions = lm.predict(X_)
        rmse_train = math.sqrt(mean_squared_error(Y_train, predictions))
        predictions = lm.predict(X_test_)
        rmse_test = math.sqrt(mean_squared_error(Y_test, predictions))
        rmse_train_list.append(rmse_train)
        rmse_test_list.append(rmse_test)
    print(deg)
    print("Average Train RMSE: ",math.fsum(rmse_train_list)/len(rmse_train_list))
    print("Average Test RMSE: ", (math.fsum(rmse_test_list)/len(rmse_test_list)))
    train_rmse.append((math.fsum(rmse_train_list)/len(rmse_train_list)))
    test_rmse.append((math.fsum(rmse_test_list)/len(rmse_test_list)))

    X_test_fin = poly.fit_transform(X_test1)
    predictions = lm.predict(X_test_fin)
    rmse = math.sqrt(mean_squared_error(Y_test1, predictions))
    holdout_rmse.append(rmse)
    print("Holdout RMSE: ", holdout_rmse)

plt.plot(degree_list, train_rmse, 'red', label='Train rmse')
plt.plot(degree_list, test_rmse, 'blue', label='Test rmse')
plt.plot(degree_list, holdout_rmse, 'green', label='Hold out rmse')
plt.xlabel('Increasing degree')
plt.ylabel('RMSE')
plt.title('Model selection plot')
plt.legend()
plt.show()

# print('Avg_train_rmse =' + str(math.fsum(rmse_train_list)/len(rmse_train_list)))
# print('Avg_test_rmse = ' + str(math.fsum(rmse_test_list)/len(rmse_test_list) ))
# print("holdout_test_rmse =" , rmse)
# plt.figure(fi
# gsize = (8,8))
# plt.scatter(Y_test1,predictions)
# plt.show()
print(train_rmse)
print(test_rmse)

# title = "Learning Curves (Linear Regression)"
# cv = ShuffleSplit(n_splits=5, test_size=0.70)
# poly = PolynomialFeatures(degree=1)
# X = poly.fit_transform(X)
# estimator = lm
#
# train_sizes=np.linspace(.1, 1.0, 5)

# plt.figure()
# plt.title(title)
# plt.xlabel("Training examples")
# plt.ylabel("RMSE SCORE")
#
# train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, Y, cv=cv, train_sizes=train_sizes,  scoring = 'neg_mean_squared_error')
#
# train_scores_mean = np.sqrt(np.mean(-train_scores, axis=1))
# # train_scores_std = np.std(-train_scores, axis=1)
# test_scores_mean = np.sqrt(np.mean(-test_scores, axis=1))
# # test_scores_std = np.std(-test_scores, axis=1)
#
# plt.grid()
#
# # plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
# #                  train_scores_mean + train_scores_std, alpha=0.1,
# #                  color="r")
# # plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
# #                  test_scores_mean + test_scores_std, alpha=0.1, color="g")
# plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#          label="Training rmse")
# plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#          label="Cross-validation rmse")
#
# plt.legend(loc="best")
#
# plt.show()


with open('LinearRegressor.pkl', 'wb') as f:
    pickle.dump(lm, f)
    print('Saved')

