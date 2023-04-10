from ETS import * 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = { 'size'   : 12}
matplotlib.rc('font', **font)
inputs = 'Data//'
outputs = 'Figures//'

color = 'C1'

########################################

name = 'chap2-KPI-fig'
df = pd.read_csv(inputs+name+'.csv')

ax = df.plot(figsize=(8,3.25),style=['-', '--',':'])
ax.set_xlabel('Period')
ax.set_ylabel('Volume')

plt.axhline(y=2,color=color,ls='-',label='Forecast #1',linewidth=1.5,zorder=0)
plt.axhline(y=4,color=color,ls='--',label='Forecast #2',linewidth=1.5,zorder=0)
plt.axhline(y=6,color=color,ls=':',label='Forecast #3',linewidth=1.5,zorder=0)
ax.legend(loc="upper right",bbox_to_anchor=(1.3, 1.0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 

plt.tight_layout()
plt.savefig(outputs+name+'.pdf')

########################################

name = 'chap2-mean-median-fig'
df = pd.read_csv(inputs+name+'.csv')

mean = (df['Demand']*df['Probability']).sum()
median = df['Demand'][(df['Probability'].cumsum() > 0.5).idxmax()]

ax = df.plot(x='Demand',y='Probability',legend=False,figsize=(8,3.25))
ax.set_xlabel('Demand')
ax.set_ylabel('Probability')

plt.axvline(x=mean,color=color,ls='--',label='Average',linewidth=1.5,zorder=0)
plt.axvline(x=median,color=color,ls=':',label='Median',linewidth=1.5,zorder=0)
ax.legend(loc="upper right",bbox_to_anchor=(1.3, 1.0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 

plt.tight_layout()
plt.savefig(outputs+name+'.pdf')