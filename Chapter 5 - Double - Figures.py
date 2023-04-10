from ETS import * 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = { 'size'   : 12}
matplotlib.rc('font', **font)
inputs = 'Data//'
outputs = 'Figures//'

########################################

name = 'chap5-double1-fig'
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


########################################


name = 'chap5-double2-fig'
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


########################################

name = 'chap5-double3-fig'
df = pd.read_excel(inputs+name+'.xlsx')

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

name = 'chap5-double-components-fig'
d = pd.read_csv(inputs+'chap5-double1-fig'+'.csv').Demand.dropna().values
df = double_exp_smooth(d,extra_periods=10)

fig, ax = plt.subplots(figsize=(8,3.25))
plot1 = ax.plot(df.index,df[['Demand']],ls='-')
plot2 = ax.plot(df.index,df[['Level']],ls='--')
ax.set_xlabel("Period")
ax.set_ylabel("Volume")
ax1 = ax.twinx()
plot3 = ax1.plot(df.index,df['Trend'],color="C2",ls="-.")
ax.legend(plot1+plot2+plot3, ['Demand','Level','Trend'],loc="best",bbox_to_anchor=(1.075, 1.0), ncol=1)
plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 
plt.tight_layout()
plt.savefig(outputs+name+'.pdf')



#######################################

d = [28,19,18,13,19,16,19,18,13,16,16,11,18,15,13,15,13,11,13,10,12]   
df = double_exp_smooth(d, extra_periods=4)
kpi(df)
df.index.name = 'Period'
df[['Demand','Forecast']].plot(figsize=(8,3),title='Double Smoothing',ylim=(0,30),style=['-','--'])  
plt.tight_layout()
plt.savefig(outputs+'chap5-double-exp-smooth-example-fig'+'.pdf')

