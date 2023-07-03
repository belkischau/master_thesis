import pandas as pd 

df = pd.read_csv("netmhcpan/MHC_dict.csv")


a = 0
proteins_list = list()
allele_list = list()

for index, row in df.iterrows(): 
    allele_list.append(row['allele'])
    protein = row['proteins'].replace("'", "")
    protein = protein.replace('"', '')
    protein = protein.replace('[', '')
    protein = protein.replace(']', '')
    protein = protein.replace(' ', '')

    w = ''
    b = 0

    for p in protein.split(','):
#        print(p)
        p = ''.join(p)
        if w == '':
            w += p
        else: 
            w = w + ',' + p
#        b += 1
#        if b == 10:
#            break
#    a += 1
#    if a == 5:
#        break
    proteins_list.append(w)

new_df = pd.DataFrame({
    'allele': allele_list,
    'proteins': proteins_list
    })

new_df.to_csv('netmhcpan/allele_protein.csv', index = None, sep = ',')
