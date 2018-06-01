import numpy as np
from sklearn.cluster import KMeans
import csv

# open the file and reading the values into df

with open('C:\\Users\\risin\\Downloads\\DS\\data_1024.csv','r') as csvfile:
    data=csv.reader(csvfile, delimiter='\t')
    i=0
    
    i=0
    df={}
    for row in data:
        if i==0:
            i=1
        else:
           key=row[0]
           df[key] = row[1:]
           # df[row[0]]=row[1]
print (df)

### For the purposes of this example, we store feature data from our
### dataframe `df`, in the `f1` and `f2` arrays. We combine this into
### a feature matrix `X` before entering it into the algorithm.
""" 
f1 = df['Distance_Feature'].values
f2 = df['Speeding_Feature'].values

X=np.matrix(zip(f1,f2))
kmeans = KMeans(n_clusters=2).fit(X) """
i=1
for key in df:
    if i==1:
        print (key)
        i=0
    else:
        pass