import pickle
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class FoodItem:
    def __init__(self, id, name, features, cluster):
        self.id = id
        self.name = name
        self.features = features
        self.cluster = cluster


class FinalCluster:
    def __init__(self, foodItem, subCluster):
        self.foodItem = foodItem
        self.subCLuster = subCluster

numberOfClusters = 8

with open('TrainedDict.pkl', 'rb') as f:
    trainedDict = pickle.load(f)

with open('LinearRegressor.pkl', 'rb') as f:
    regressor = pickle.load(f)

with open('myDict.pkl', 'rb') as f:
    myDict = pickle.load(f)

print('***** Welcome to Smart Health System, Author: Akshita Sawhney - MT17143 *****')
flag = '0'
while flag != '3':
    print('Type 1 for predicting calorific value using trained regressor')
    print('Type 2 for using recommendation system built using Self Organizing Maps and K-Means clustering')
    print('Type 3 to quit!')
    flag = input()
    if flag == '1':
        print('Please enter protein content per 100grams in grams')
        protein = float(input())
        print('Please enter fat content per 100grams in grams')
        fat = float(input())
        print('Please enter carbs content per 100grams in grams')
        carbs = float(input())
        print('Please enter fiber content per 100grams in grams')
        fiber = float(input())
        print('Please enter vitamin c content per 100grams in grams')
        vitc = float(input())
        myListOfFeatures = [protein, fat, carbs, fiber, vitc]
        poly = PolynomialFeatures(degree=1)
        myListOfFeatures = poly.fit_transform(np.asarray(myListOfFeatures).reshape(1,-1))
        predicted_energy = regressor.predict(myListOfFeatures)
        print(str(predicted_energy[0]) + ' kcal')


    elif flag == '2':
        print('Please select a food category by typing its category number')
        for i in range(numberOfClusters):
            print('Category ' + str(i))
        cat = int(input())
        if cat in trainedDict:
            listOfFoodItems = trainedDict[cat]
            for f in listOfFoodItems:
                print('Id:' + str(f.foodItem.id) + ', Name:' + str(f.foodItem.name))
            print('Please enter the food id of the selected food item')
            selectedId = input()
            selectedObj = None
            for f in listOfFoodItems:
                if selectedId == str(f.foodItem.id):
                    print('Selected item details')
                    print('Name-' + str(f.foodItem.name))
                    print('Energy, Protein, Fat, Carbohydrates, Fiber, Vitamin C')
                    print('Nutrients-' + str(f.foodItem.features))
                    selectedObj = f
                    selectedEnergy = myDict[int(selectedId)]
                    print('Calories - unnormalized = ' + str(selectedEnergy))
                    break
            subCluster = selectedObj.subCLuster
            flg = 0
            if subCluster == 0 or subCluster == 1:
                print('You have selected a healthy food item!')
            elif subCluster == 2:
                print('You have selected a moderately unhealthy item')
                flg = 1
            elif subCluster == 3:
                print('You have selected a highly unhealthy item')
                flg = 1
            print('Our 5 recommendations are as follows:')
            if flg == 1:
                c = 0
                for f in listOfFoodItems:
                    if f.subCLuster == 0 or f.subCLuster == 1:
                        if myDict[int(f.foodItem.id)] <= selectedEnergy:
                            print('Id-' + str(f.foodItem.id))
                            print('Name-' + str(f.foodItem.name))
                            print('Energy, Protein, Fat, Carbohydrates, Fiber, Vitamin C')
                            print('Nutrients-' + str(f.foodItem.features))
                            print('Calories - unnormalized = ' + str(myDict[int(f.foodItem.id)]))
                            c += 1
                    if c == 5:
                        break
    elif flag == '3':
        pass
    else:
        print('Please enter valid input!')
print('Thank You!')
#3
#45145408