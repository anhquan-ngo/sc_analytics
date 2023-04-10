import pandas as pd
import numpy as np
from scipy.stats import norm

###################################################

# WINSORIZATION
#higher_limit = np.percentile(array, 99).round(1)
#lower_limit = np.percentile(array, 1).round(1)
#array = np.clip(array, a_min=lower_limit, a_max=higher_limit)

###################################################

# NORMALIZATION
df = pd.read_csv('Data//chap10-outliers1-fig.csv',index_col = 0)
m = df.mean()
s = df.std()
limit_high = norm.ppf(0.99,m,s)
limit_low = norm.ppf(0.01,m,s)
df = df.clip(lower=limit_low, upper=limit_high, axis=1)


#Print the probabilities of each demand observation
print(norm.cdf(df.values, m, s).round(2))

###################################################

df = pd.read_csv('Data//chap10-outliers4-fig.csv',index_col =0)

df['Error'] = df['Forecast'] - df['Demand']
m = df['Error'].mean()
s = df['Error'].std()

limit_high = norm.ppf(0.99,m,s) + df['Forecast']
limit_low = norm.ppf(0.01,m,s) + df['Forecast']
df['Updated'] = df['Demand'].clip(lower=limit_low,upper=limit_high).round(0).astype(int)
print(df)

###################################################

df['Error'] = df['Forecast'] - df['Demand']
m = df['Error'].mean()
s = df['Error'].std()

prob = norm.cdf(df['Error'], m, s)
outliers = (prob > 0.99) | (prob < 0.01)

m2 = df.loc[~outliers,'Error'].mean()
s2 = df.loc[~outliers,'Error'].std()

limit_high = norm.ppf(0.99,m2,s2) + df['Forecast']
limit_low = norm.ppf(0.01,m2,s2) + df['Forecast']
df['Updated'] = df['Demand'].clip(lower=limit_low, upper=limit_high).round(0).astype(int)