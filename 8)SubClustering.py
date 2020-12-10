import pickle

# Energy, Protein, Fat, Carbs, Fiber, VitC


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

with open('ClusteredDict.pkl', 'rb') as f:
    clusteredDict = pickle.load(f)

with open('myDataFrameForSOM.pkl', 'rb') as f:
    myDataFrameForSOM = pickle.load(f)

Protein = myDataFrameForSOM['Protein']
Fat = myDataFrameForSOM['Total lipid (fat)']
Carbs = myDataFrameForSOM['Carbohydrate, by difference']
minProtein = Protein.min()
maxProtein = Protein.max()
minFat = Fat.min()
maxFat = Fat.max()
minCarbs = Carbs.min()
maxCarbs = Carbs.max()
proteinLimit = minProtein + (maxProtein-minProtein)/3
fatLimit = minFat + (maxFat-minFat)/3
carbsLimit = minCarbs + (maxCarbs-minCarbs)/3

finalDict = dict()
for k, v in clusteredDict.items():
    for obj in v:
        id = obj.id
        name = obj.name
        features = obj.features
        protein = features[1]
        fat = features[2]
        carbs = features[3]
        counter = 0
        if protein < proteinLimit:
            counter += 1
        if fat >= fatLimit:
            counter += 1
        if carbs >= carbsLimit:
            counter += 1
        subcat = counter
        finalFoodItem = FinalCluster(obj, subcat)
        if k in finalDict:
            listOfObjects = finalDict[k]
            listOfObjects.append(finalFoodItem)
            finalDict[k] = listOfObjects
        else:
            listOfObjects = [finalFoodItem]
            finalDict[k] = listOfObjects
with open('TrainedDict.pkl', 'wb') as f:
    pickle.dump(finalDict, f)
print('Done')
