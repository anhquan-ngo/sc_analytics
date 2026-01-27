from ETS import * 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = { 'size'   : 12}
matplotlib.rc('font', **font)
inputs = 'Data//'
outputs = 'Figures//'

#######################################

name = 'chap7-double-damped-fig'
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

#######################################

d=[28,19,18,13,19,16,19,18,13,16,16,11,18,15,13,15,13,11,13,10,12]   
df = double_exp_smooth_damped(d,extra_periods=4)
kpi(df)
df.index.name = 'Period'
df[['Demand','Forecast']].plot(figsize=(8,3),title='Damped Double Smoothing',ylim=(0,30),style=['-','--'])  
plt.tight_layout()
plt.savefig(outputs+'chap7-damped-exp-smooth-example-fig'+'.pdf')
