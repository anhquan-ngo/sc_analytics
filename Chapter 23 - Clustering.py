import numpy as np
import pandas as pd
from support import import_data
import matplotlib.pyplot as plt

#####################################################################

def seasonal_factors(df,slen):
    s = pd.DataFrame(index=df.index)
    for i in range(slen):
        idx = [x for x in range(df.shape[1]) if x%slen==i] # Indices that correspond to this season
        s[i+1] = np.mean(df.iloc[:,idx],axis=1)       #+1 to put the right month number
    s = s.divide(s.mean(axis=1),axis=0).fillna(0)
    return s

def scaler(s):
    mean = s.mean(axis=1)
    maxi = s.max(axis=1)
    mini = s.min(axis=1)
    s = s.subtract(mean,axis=0)
    s = s.divide(maxi-mini,axis=0).fillna(0)
    return s

df = import_data()
from sklearn.cluster import KMeans 
s = seasonal_factors(df,slen=12)
s = scaler(s)
kmeans = KMeans(n_clusters=4, random_state=0).fit(s)
df['Group'] = kmeans.predict(s)

#####################################################################

results = []
for n in range(1,10):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(s)
    results.append([n, kmeans.inertia_])
results = pd.DataFrame(data=results,columns=['Number of clusters','Intertia']).set_index('Number of clusters')
results.plot()
plt.show()

#####################################################################

import calendar
kmeans = KMeans(n_clusters=4, random_state=0).fit(s)
centers = pd.DataFrame(data=kmeans.cluster_centers_).transpose()
centers.index = calendar.month_abbr[1:]
centers.columns = [f'Cluster {x}' for x in range(centers.shape[1])]

import seaborn as sns
sns.heatmap(centers, annot=True, fmt='.2f', center=0, cmap='RdBu_r')
plt.show()

#####################################################################
print(df['Group'].value_counts().sort_index())
