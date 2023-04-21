from ETS import * 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = { 'size'   : 12}
matplotlib.rc('font', **font)
inputs = 'Data//'
outputs = 'Figures//'

############################################

name = 'chap1-moving-average-fig'
df = pd.read_csv(inputs+name+'.csv')

ax = df.plot(figsize=(8,3.5),style=['-', '--',':'])
ax.set_xlabel('Period')
ax.set_ylabel('Volume')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 

plt.tight_layout()
plt.savefig(outputs+name+'.pdf')

############################################


d=[28,19,18,13,19,16,19,18,13,16,16,11,18,15,13,15,13,11,13,10,12]   
df = moving_average(d, extra_periods=4, n=3)
df.index.name = 'Period'
df[['Demand','Forecast']].plot(figsize=(8,3),title='Moving average',ylim=(0,30),style=['-','--'])  
MAE = df['Error'].abs().mean()  
print('MAE:',round(MAE,2)) 
RMSE = np.sqrt((df['Error']**2).mean())
print('RMSE:',round(RMSE,2))
plt.tight_layout()
plt.savefig(outputs+'chap1-moving-average-example'+'.pdf')
