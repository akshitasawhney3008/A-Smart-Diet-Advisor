import numpy as np
import pandas as pd
import math
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVR
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA
# import seaborn as sns

my_file  = np.genfromtxt('nutrients_train_data.tsv',dtype='unicode',delimiter='\t')
my_file = my_file.astype(np.float)
np.random.shuffle(my_file)


X = my_file[:,:-1]
Y = my_file[:,-1]
Y = Y.astype(np.float)


X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X,Y,test_size=0.25)


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
    lm = LinearSVR(C= 0.4)
    # poly = PolynomialFeatures(degree=deg)
    # degree_list.append(deg)
    train_data = np.append(X_train1,Y_train1,axis=1)
    for train,test in kf.split(train_data):
        X_train = train_data[train][:,:-1]
        X_test = train_data[test][:,:-1]
        Y_train = train_data[train][:,-1]
        Y_test = train_data[test][:,-1]
        # X_ = poly.fit_transform(X_train)
        # X_test_ = poly.fit_transform(X_test)
        lm = lm.fit(X_train,Y_train)
        # lm = lm.fit(X_, Y_train)
        predictions = lm.predict(X_train)
        rmse_train = math.sqrt(mean_squared_error(Y_train, predictions))
        predictions = lm.predict(X_test)
        rmse_test = math.sqrt(mean_squared_error(Y_test, predictions))
        rmse_train_list.append(rmse_train)
        rmse_test_list.append(rmse_test)
    print(deg)
    print("Average Train RMSE: ",math.fsum(rmse_train_list)/len(rmse_train_list))
    print("Average Test RMSE: ", (math.fsum(rmse_test_list)/len(rmse_test_list)))
    train_rmse.append((math.fsum(rmse_train_list)/len(rmse_train_list)))
    test_rmse.append((math.fsum(rmse_test_list)/len(rmse_test_list)))

    # X_test_fin = poly.fit_transform(X_test1)
    predictions = lm.predict(X_test1)
    rmse = math.sqrt(mean_squared_error(Y_test1, predictions))
    holdout_rmse.append(rmse)
    print("Holdout RMSE: ", holdout_rmse)

# plt.plot(degree_list, train_rmse, 'red', label='Train rmse')
# plt.plot(degree_list, test_rmse, 'blue', label='Test rmse')
# plt.plot(degree_list, holdout_rmse, 'green', label='Hold out rmse')
# plt.xlabel('Increasing degree')
# plt.ylabel('RMSE')
# plt.title('Model selection plot')
# plt.legend()
# plt.show()

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
# cv = ShuffleSplit(n_splits=5, test_size=0.2)
# poly = PolynomialFeatures(degree=1)
# X = poly.fit_transform(X)
# estimator = lm
#
# train_sizes=np.linspace(.1, 1.0, 8)
#
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


# with open('LinearRegressor.pkl', 'wb') as f:
#     pickle.dump(lm, f)
#     print('Saved')

