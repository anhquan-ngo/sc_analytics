import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
font = { 'size'   : 12}
matplotlib.rc('font', **font)
inputs = 'Data//'
outputs = 'Figures//'

########################################

# name = 'chap4-underfit1-fig'
# df = pd.read_csv(inputs+name+'.csv')

# ax = df.plot(figsize=(8,3.5),style=['-', '--',':'])
# ax.set_xlabel('Period')
# ax.set_ylabel('Volume')
# ax.legend(loc="best",bbox_to_anchor=(1.2, 1.0))
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
#                 labelbottom=False,labelleft=False) 
# plt.tight_layout()
# plt.savefig(outputs+name+'.pdf')

########################################

name = 'chap4-underfit2-fig'
df = pd.read_csv(inputs+name+'.csv')

ax = df.plot(figsize=(8,3.25),style=['-', '--',':'])
ax.set_xlabel('Period')
ax.set_ylabel('Volume')
ax.legend(loc="best",bbox_to_anchor=(1.0, 1.0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 

plt.tight_layout()
plt.savefig(outputs+name+'.pdf')

########################################

name = 'chap4-underfit3-fig'
df = pd.read_csv(inputs+name+'.csv')

ax = df.plot(figsize=(8,3.25),style=['-', '--',':'])
ax.set_xlabel('Period')
ax.set_ylabel('Volume')

ax.legend(loc="best",bbox_to_anchor=(1.0, 1.0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 

plt.tight_layout()
plt.savefig(outputs+name+'.pdf')

########################################

name = 'chap4-underfit4-fig'
df = pd.read_csv(inputs+name+'.csv')

ax = df.plot(figsize=(8,3.25),style=['-', '--',':'])
ax.set_xlabel('Period')
ax.set_ylabel('Volume')

ax.legend(loc="best",bbox_to_anchor=(1.0, 1.0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 

plt.tight_layout()
plt.savefig(outputs+name+'.pdf')
