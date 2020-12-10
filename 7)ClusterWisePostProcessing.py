import pickle


class FoodItem:
    def __init__(self, id, name, features, cluster):
        self.id = id
        self.name = name
        self.features = features
        self.cluster = cluster


with open('myDataFrame.pkl', 'rb') as f:
    myDataFrame = pickle.load(f)
productNameDataSeries = myDataFrame['Product Name']
ids = myDataFrame['NDB_No']

# Replace the list of food items with the actual list

with open('data_with_clusterid.pkl', 'rb') as c:
    listOfFoodItems = pickle.load(c)
# listOfFoodItems = [[0.1, 0.2, 0.3, 0.4, 0], [0.5, 0.6, 0.7, 0.8, 1], [0.9, 1.0, 1.1, 1.2, 2], [1.3, 1.4, 1.5, 1.6, 3]]

clusterWIseDict = dict()
numberOfFoodItems = len(listOfFoodItems)
numberOfFeatures = len(listOfFoodItems[0])-1
for idx in range(numberOfFoodItems):
    foodItemList = listOfFoodItems[idx]
    features = foodItemList[0: 0+numberOfFeatures]
    cluster = foodItemList[-1]
    name = productNameDataSeries[idx]
    id = ids[idx]
    foodItemObj = FoodItem(id, name, features, cluster)
    if cluster in clusterWIseDict:
        listOfObjects = clusterWIseDict[cluster]
        listOfObjects.append(foodItemObj)
        clusterWIseDict[cluster] = listOfObjects
    else:
        listOfObjects = [foodItemObj]
        clusterWIseDict[cluster] = listOfObjects

with open('ClusteredDict.pkl', 'wb') as f:
    pickle.dump(clusterWIseDict, f)
print('Done')
