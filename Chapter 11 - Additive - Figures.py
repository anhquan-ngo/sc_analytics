from ETS import * 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
font = { 'size'   : 12}
matplotlib.rc('font', **font)
inputs = "Data//"
outputs = "Figures//"

########################################

name = "chap11-triple-add1-fig"
df = pd.read_csv(inputs+name+".csv")

ax = df.plot(figsize=(8,3.5),style=['-', '--',":"])
ax.set_xlabel("Period")
ax.set_ylabel("Volume")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc="upper center",bbox_to_anchor=(0.5, 1.2), ncol=3)
plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 

plt.tight_layout()
plt.savefig(outputs+name+".pdf")

########################################

name = "chap11-triple-add2-fig"
df = pd.read_csv(inputs+name+".csv")

ax = df.plot(figsize=(8,3.5),style=['-', '--',":"])
ax.set_xlabel("Period")
ax.set_ylabel("Volume")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 

plt.tight_layout()
plt.savefig(outputs+name+".pdf")