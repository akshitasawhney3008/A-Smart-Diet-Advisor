import pandas as pd
import pickle


fileName = 'final_normalized_nut_data.tsv'
myDataFrame = pd.read_csv(fileName, sep='\t')
myDataFrame.drop('Serving Size', axis=1, inplace=True)
myDataFrame.drop(myDataFrame.columns[len(myDataFrame.columns)-1], axis=1, inplace=True)
maxEnergy = myDataFrame['Energy'].max()
Id = myDataFrame['NDB_No']
Energy = myDataFrame['Energy']

myDict = dict()
for i in range(myDataFrame.shape[0]):
    myDict[Id[i]] = Energy[i]
with open('myDict.pkl', 'wb') as f:
    pickle.dump(myDict, f)