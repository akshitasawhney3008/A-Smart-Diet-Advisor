import numpy as np
import pandas as pd

def my_split(my_string):
    tail = my_string.lstrip('0123456789.')
    head = my_string[:len(my_string)-len(tail)]
    return head,tail



my_file = open('nutrients_data_with_energy.tsv','r')

flag = 0
rows_with_na_value = []
rows_without_na = []

nutrients_data = []
for line in my_file:
    my_nut_array = line.split('\t')
    nutrients_data.append(my_nut_array)

nutrients_array = np.asarray(nutrients_data)

new_cols = []
for i in range(nutrients_array.shape[1]):
    if i == 0 or i == 1 or i == 2:
        new_cols.append(nutrients_array[:,i].tolist())
    else:
        col = nutrients_array[:,i]
        new_val_col = []
        for j in range(0,len(col)):
            if j == 0:
                new_val_col.append(col[j])
            else:
                if col[j] == 'N.A':
                    new_val_col.append(col[j])
                elif col[j] == 'N.A\n':
                    new_val_col.append(col[j])
                else:
                    head, tail = my_split(col[j])
                    if tail == 'mg':
                        head = float(head) / 1000    # get everything in grams
                    elif tail == 'IU':
                        head = float(head) / 3.33
                        head = head / 1000000
                    new_val_col.append(float(head))

        # header = new_val_col.pop(0)
        # new_val_col = np.asarray(new_val_col)
        # new_val_col = np.reshape(new_val_col,(new_val_col.shape[0],1))
        # # new_val_col = new_val_col.transpose()
        # mean_row = np.mean(new_val_col,axis=0)
        # std_row = np.std(new_val_col,axis=0)
        # new_val_col = (new_val_col - np.mean(new_val_col,axis=0)) / np.mean(new_val_col,axis=0)
        # new_val_col= new_val_col.tolist()
        mean_nc = 0
        nc = []
        for c in range(1,len(new_val_col)):
            if new_val_col[c] == 'N.A':
                continue
            elif new_val_col[c] == 'N.A\n':
                continue
            else:
                nc.append(float(new_val_col[c]))
        # header = nc.pop(0)
        mean_nc = sum(nc)/len(nc)

        nc = []
        for c in new_val_col:
            if c == 'N.A':
               nc.append(mean_nc)
            elif c == 'N.A\n':
                nc.append(mean_nc)
            else:
                nc.append(c)

        new_cols.append(nc)

#normalize the data
new_cols = np.asarray(new_cols).transpose()
new_cols_1 = new_cols[1:,3:]
new_cols_1 = new_cols_1.astype(np.float)
# new_cols_1 = (new_cols_1 - np.mean(new_cols_1,axis=0)) / np.std(new_cols_1,axis=0)
Y = new_cols_1[:,11]
Y = np.reshape(Y,(Y.shape[0],1))
new_cols_1 = np.delete(new_cols_1,11,axis=1)
new_cols_1 =  (new_cols_1.astype(float) / new_cols_1.astype(float).max(axis=0))
new_cols_1 = np.append(new_cols_1,Y,axis=1)
np.savetxt("nutrients_train_data.tsv",new_cols_1, delimiter='\t')
row = new_cols[0,:]
row = np.reshape(row,(1,row.shape[0]))
col_to_add = new_cols[1:,:3]
new_cols_1 = np.append(col_to_add,new_cols_1,axis=1)
new_cols_1 = np.append(row,new_cols_1,axis=0)

new_cols = new_cols_1.tolist()
headers = new_cols.pop(0)
df = pd.DataFrame(new_cols, columns=headers)
df.to_csv("final_normalized_nut_data.tsv", sep='\t', encoding='utf-8')