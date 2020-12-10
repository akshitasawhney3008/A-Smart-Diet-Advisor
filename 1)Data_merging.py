import pandas as pd

def reinitialize_dict(dict1):
    for key,value in dict1.items():
        dict1[key] = 'N.A'
    return dict1

nutrients_file = open('Nutrients.txt','r')
product_file = open('Products.csv', 'r' , encoding='utf8')
serving_size_file = open('Serving_size.csv','r',encoding='utf-8')
nutrients_file_read = nutrients_file.readlines()
product_file_read = product_file.readlines()
serving_size_file_read = serving_size_file.readlines()

flag = 0
column_names = []

dict_of_nutrients = {}

for line in nutrients_file_read:
    nut_f = line.split('\t')
    if nut_f[2] == 'Nutrient_name':
        column_names.append(nut_f[0])
        column_names.append("Product Name")
        column_names.append("Serving Size")
    elif nut_f[2] not in dict_of_nutrients:
        dict_of_nutrients[nut_f[2]] = 'N.A'
        column_names.append(nut_f[2])



dict_product_names = {}
for prod in product_file_read:
    prod_list = prod.split(',')
    dict_product_names[prod_list[0].lstrip("\"").rstrip("\"")] = prod_list[1].lstrip("\"").rstrip("\"")

dict_of_serving_size = {}
for ss in serving_size_file_read:
    ss_list = ss.split(',')
    dict_of_serving_size[ss_list[0].lstrip("\"").rstrip("\"")] = str(ss_list[1].lstrip("\"").rstrip("\"") + ss_list[2].lstrip("\"").rstrip("\""))

final_nut_data = []
final_nut_data.append(column_names)
i = 1
while i < len(nutrients_file_read):
    rows = []
    nutri = nutrients_file_read[i].split('\t')
    rows.append(nutri[0])
    rows.append(dict_product_names[nutri[0]])
    rows.append(dict_of_serving_size[nutri[0]])
    ndb_no = nutri[0]
    j = i
    dict_of_nutrients = reinitialize_dict(dict_of_nutrients)
    while j < len(nutrients_file_read) and nutrients_file_read[j].split('\t')[0] == ndb_no:
        nutri_j = nutrients_file_read[j].split('\t')
        if nutri_j[2] in dict_of_nutrients:
            dict_of_nutrients[nutri_j[2]] = nutri_j[4] + nutri_j[5].rstrip('\n')
            j = j+1
    for key,value in dict_of_nutrients.items():
        rows.append(value)
    final_nut_data.append(rows)
    i = j

headers = final_nut_data.pop(0)
df = pd.DataFrame(final_nut_data, columns=headers)
df.to_csv("final_nutrients_data.tsv", sep='\t', encoding='utf-8')
