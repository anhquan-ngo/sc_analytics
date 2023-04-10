from ETS import * 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = { 'size'   : 12}
matplotlib.rc('font', **font)
inputs = 'Data//'
outputs = 'Figures//'

####################################

name = 'chap3-level-fig'

def moving_average(a, n=7) :
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

x = np.arange(100)
delta = np.random.uniform(-10,10, size=(100,))
ts = .4 * x +3 + delta
# df = double_exp_smooth(ts,extra_periods=1,alpha=0.2,beta=0.2)
df = pd.DataFrame([ts[2:-3],moving_average(ts)]).T
df.columns = ['Demand','Level']

ax = df[['Demand','Level']].plot(figsize=(8,3.5),style=['-', '--'])
# ax = df[['Level','Demand']].plot(figsize=(8,3.5),style=['--','-'],color=['C1','C0'])
ax.set_xlabel('Period')
ax.set_ylabel('Volume')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 
plt.tight_layout()
plt.savefig(outputs+name+'.pdf')

###############################

name = 'chap3-simple-fig'
df = pd.read_csv(inputs+name+'.csv')
df1 = simple_exp_smooth(df.Demand.dropna()[:-5], extra_periods=5, alpha=0.1)
dic = {'Forecast':r"Forecast ($\alpha=0.1$)"}
ax = df1[["Demand","Forecast"]].rename(columns=dic).plot(figsize=(8,3.5),style=['-', '--'])
df2 = simple_exp_smooth(df.Demand.dropna()[:-5], extra_periods=5, alpha=0.8)
dic = {'Forecast':r"Forecast ($\alpha=0.8$)"}
df2[["Forecast"]].rename(columns=dic).plot(figsize=(8,3.5),style=[':'],color='C2',ax=ax)

# ax = df.plot(figsize=(8,3.5),style=['-', '--',':'])

ax.set_xlabel('Period')
ax.set_ylabel('Volume')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

x_point = 51
y_point = df1['Forecast'][x_point]
plt.plot(x_point,y_point, marker='o',color='C1',markersize=4)
ax.text(x_point,y_point+3, r'$f_{t^*}$', size=13, color='C1', va='bottom',ha='center')
y_point = df2['Forecast'][x_point]
plt.plot(x_point,y_point, marker='o',color='C2',markersize=4)
ax.text(x_point,y_point-3, r'$f_{t^*}$', size=13, color='C2', va='top',ha='center')
xticks = ['$t^*$']
plt.xticks([x_point], xticks)

plt.tick_params(axis='both',which='both',bottom=True,top=False,left=False,
                labelbottom=True,labelleft=False) 
plt.tight_layout()
plt.savefig(outputs+name+'.pdf')

###############################♦

d=[28,19,18,13,19,16,19,18,13,16,16,11,18,15,13,15,13,11,13,10,12]   
df = simple_exp_smooth(d, extra_periods=4)
kpi(df)
df.index.name = 'Period'
df[['Demand','Forecast']].plot(figsize=(8,3),title='Simple Smoothing',ylim=(0,30),style=['-','--'])  
plt.tight_layout()
plt.savefig(outputs+'chap3-simple-exp-smooth-example-fig'+'.pdf')


###############################♦

d=[28,19,18,13,19,16,19,18,15,16,16,11,18,15,13,14,13,11,9,10,8]   
df = simple_exp_smooth(d, extra_periods=4)
df.index.name = 'Period'
ax = df[['Demand','Forecast']].plot(figsize=(8,2.5),ylim=(4,30),xlim=(-1,25),style=['-','--'])  
ax.legend(loc="best",bbox_to_anchor=(1, 1), ncol=1)
plt.plot(21,df['Forecast'][21], marker='x',color='C1',markersize=8,mew=2)
ax.text(21, 12, r'$f_{t^*}$', size=13, color='C1', va='center',ha='center')
plt.annotate(s='', xy=(21,8), xytext=(24,8), arrowprops={"arrowstyle":'<-',"ls":"-","lw":1,"color":"C1"})
ax.text(22.5, 12, r'$f_{t>t^*}$', size=13, color='C1', va='center',ha='center')

# plt.annotate(s='', xy=(21,4), xytext=(25,4), 
#              arrowprops={"arrowstyle":'<-',"ls":"-","lw":1,"color":"k"})
# ax.text(22.5, 6, r'$t>t^*$', size=13, color='k', va='center',ha='center')
ax.set_ylabel("Volume")
xticks = ['$t^*$']
plt.xticks([21], xticks)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='both',which='both',bottom=True,top=False,left=False,
                labelbottom=True,labelleft=False) 
plt.tight_layout()
plt.savefig(outputs+'chap3-future-forecast-fig'+'.pdf')