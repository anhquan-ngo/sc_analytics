from ETS import * 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = { 'size'   : 12}
matplotlib.rc('font', **font)
inputs = "Data//"
outputs = "Figures//"

########################################

name = "chap9-triple-mul1-fig"
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

########################################

name = "chap9-triple-mul2-fig"
df = pd.read_csv(inputs+name+".csv")
# df.columns = ["Seasonal Factors"]

ax = df.plot(figsize=(8,3.5),style=['-', '--',":"])
ax.set_xlabel("Period")
ax.set_ylabel("Multiplicator")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 

plt.tight_layout()
plt.savefig(outputs+name+".pdf")

########################################

name = "chap9-triple-mul3-fig"
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

########################################

name = "chap9-triple-mul4-fig"
df = pd.read_csv(inputs+name+".csv")

ax = df.plot(figsize=(8,3.5),style=['-', '--',":"])
ax.set_xlabel("Period (quarters)")
ax.set_ylabel("Volume")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 

plt.tight_layout()
plt.savefig(outputs+name+".pdf")

########################################

# name = "chap9-triple-mul5-fig"
# df = pd.read_csv(inputs+name+".csv")
# ax = df.plot(figsize=(8,3.5),style=['-', '--'],secondary_y=["Seasonality"])
# ax.set_xlabel("Period (quarters)")
# ax.set_ylabel("Volume")
# plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
#                 labelbottom=False,labelleft=False) 
# plt.tight_layout()
# plt.savefig(outputs+name+".pdf")
# name = "chap9-triple-mul5-fig"
# df = pd.read_csv(inputs+name+".csv")

name = "chap9-triple-mul5-fig"
df = pd.read_csv(inputs+name+".csv")
fig, ax = plt.subplots(figsize=(8,3.5))
plot1 = ax.plot(df.index,df['Level'])#df['Level'].plot(ax=ax,style=['-'])
ax.set_xlabel("Period (quarters)")
ax.set_yticks([5,6,7,8,9,10,11])
ax.set_ylabel("Volume")
ax1 = ax.twinx()
plot2 = ax1.plot(df.index,100*df['Seasonality'],color="C1",ls="--")#df['Seasonality'].plot(ax=ax1,style=['--'],color="C1")
import matplotlib.ticker as mtick
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.legend(plot1+plot2, ["Level","Seasonality (right)"],loc="upper center",bbox_to_anchor=(0.5, 1.2), ncol=2)
plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 

plt.tight_layout()
plt.savefig(outputs+name+".pdf")


########################################

name = "chap9-triple-mul4-fig"
df = pd.read_csv(inputs+name+".csv")
ts = df.Demand.values[:-4]
df = triple_exp_smooth_mul(ts, slen=12, extra_periods=4, alpha=0.3, beta=0.2, phi=0.9, gamma=0.2)

name = "chap9-multi-graph-mess-fig"
df.plot(figsize=(8,4))
plt.tight_layout()
plt.savefig(outputs+name+".pdf")

name = "chap9-multi-graph-subplots-fig"
df.plot(subplots=True,figsize=(8,7))
plt.tight_layout()
plt.savefig(outputs+name+".pdf")

name = "chap9-multi-graph-two-axis-fig"
df[["Level","Trend","Season"]].plot(figsize=(8,4),secondary_y=["Season"])
plt.tight_layout()
plt.savefig(outputs+name+".pdf")
