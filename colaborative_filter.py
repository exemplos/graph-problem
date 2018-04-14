import pandas as pd
from scipy.spatial.distance import cosine


data = pd.read_csv('lastfm-matrix-germany.csv')
data_germany = data.drop('user', 1)
data_ibs = pd.DataFrame(index=data_germany.columns,columns=data_germany.columns)

print(data_ibs)

for i in range(0,len(data_ibs.columns)) :
    for j in range(0,len(data_ibs.columns)) :
      data_ibs.ix[i,j] = 1-cosine(data_germany.ix[:,i],data_germany.ix[:,j])

data_neighbours = pd.DataFrame(index=data_ibs.columns,columns=range(1,11))

for i in range(0,len(data_ibs.columns)):
    data_neighbours.ix[i,:10] = data_ibs.ix[0:,i].sort_values(ascending=False)[:10].index


print(data_neighbours.head(6).ix[:6,2:4])