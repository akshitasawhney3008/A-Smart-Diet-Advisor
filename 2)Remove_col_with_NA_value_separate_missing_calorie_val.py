import pandas as pd
import numpy as np
my_file = open('final_nutrients_data.tsv','r')

flag = 0
rows_with_na_value = []
rows_without_na = []

nutrients_data = []
for line in my_file:
    my_nut_array = line.split('\t')
    nutrients_data.append(my_nut_array)

nutrients_array = np.asarray(nutrients_data)
column_to_be_removed = []
for i in range(nutrients_array.shape[1]):
    column_data = nutrients_array[:,i]
    count = 0
    for val in column_data:
        if val == 'N.A' or val == 'N.A\n':
            count = count + 1
    if count >= 0.8*nutrients_array.shape[0]:
        column_to_be_removed.append(i)

nutrients_array = np.delete(nutrients_array, column_to_be_removed, 1)

for i in range(nutrients_array.shape[0]):
    my_nut_array = nutrients_array[i,:].tolist()
    if flag == 0:
        if 'Energy' in my_nut_array:
            index_of_energy = my_nut_array.index('Energy')
            headers = my_nut_array
            flag = 1

    else:
        if my_nut_array[index_of_energy] == 'N.A':
            rows_with_na_value.append(my_nut_array)
        else:
            rows_without_na.append(my_nut_array)


df_with_na = pd.DataFrame(rows_with_na_value, columns=headers)
df_with_na.to_csv("nutrients_data_without_energy.tsv", sep='\t', encoding='utf-8')

df_without_na = pd.DataFrame(rows_without_na, columns=headers)
df_without_na.to_csv("nutrients_data_with_energy.tsv", sep='\t', encoding='utf-8')

