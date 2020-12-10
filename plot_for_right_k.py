import matplotlib.pyplot as plt
import numpy as np

my_file = open('k_vs_index.csv', 'r')
my_file_read = my_file.readlines()

X_list = []
Y_list = []
for lines in my_file_read:
    line = lines.rstrip('\n').split(',')
    X_list.append(float(line[0]))
    Y_list.append(float(line[1]))

plt.plot(X_list, Y_list, 'red')
plt.xlabel('Values for K')
plt.ylabel('DaviesBouldin Index')
plt.title('K vs DaviesBouldin_Index')
plt.legend()
plt.show()