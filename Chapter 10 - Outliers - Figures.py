import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = { 'size'   : 12}
matplotlib.rc('font', **font)
inputs = "Data//"
outputs = "Figures//"

##############################################

import scipy.stats as stats
from scipy.stats import norm

##############################################

def winVSnor_fig(df,name):
    ax = df.plot(style=["-",":","--"],color=["steelblue"],figsize=(8,3))
    plt.axhline(y=limit_nor_high,color="firebrick",ls="--",label='Normalization',linewidth=2)
    plt.axhline(y=limit_nor_low,color="firebrick",ls="--",linewidth=2)
    plt.axhline(y=limit_win_high,color="firebrick",ls="-",label='Winsorization',linewidth=2)
    plt.axhline(y=limit_win_low,color="firebrick",ls="-",linewidth=2)
    # ax.legend(loc="lower right")
    ax.legend(loc="best",bbox_to_anchor=(1, 1), ncol=1)
    ax.set_ylabel("Volume")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                    labelbottom=False,labelleft=False) 
    plt.tight_layout()
    plt.savefig(outputs+name+".pdf")

def winsor_fig(df,name):
    ax = df.plot(style=["-",":","--"],color=["steelblue"],figsize=(8,3))
    plt.axhline(y=limit_win_high,color="firebrick",ls="-",label='Winsorization',linewidth=2)
    plt.axhline(y=limit_win_low,color="firebrick",ls="-",linewidth=2)
    # ax.legend(loc="lower right")
    ax.legend(loc="best",bbox_to_anchor=(1, 1), ncol=1)
    ax.set_ylabel("Volume")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                    labelbottom=False,labelleft=False) 
    plt.tight_layout()
    plt.savefig(outputs+name+".pdf")

name = "chap10-outliers1-fig"
df = pd.read_csv(inputs+name+".csv",index_col = 0)
df.index.name = "Period"
limit_win_high = np.percentile(df,99)
limit_win_low = np.percentile(df,1)
m = df.values.mean()
s = df.values.std()
limit_nor_high = norm.ppf(0.99,m,s)
limit_nor_low = norm.ppf(0.01,m,s)

name = "chap10-winsor1-fig"
winsor_fig(df,name)
name = "chap10-winVSnor1-fig"
winVSnor_fig(df,name)

name = "chap10-outliers2-fig"
df = pd.read_csv(inputs+name+".csv",index_col = 0)
df.index.name = "Period"

limit_win_high = np.percentile(df,99)
limit_win_low = np.percentile(df,1)
m = df.values.mean()
s = df.values.std()
limit_nor_high = norm.ppf(0.99,m,s)
limit_nor_low = norm.ppf(0.01,m,s)

name = "chap10-winsor2-fig"
winsor_fig(df,name)
name = "chap10-winVSnor2-fig"
winVSnor_fig(df,name)

###################################################

name = "chap10-outliers4-fig"
df = pd.read_csv(inputs+name+".csv",index_col = 0)
df.index.name = "Period"

df["Error"] = df["Forecast"] - df["Demand"]
m = df["Error"].mean()
s = df["Error"].std()

limit_high = norm.ppf(0.95,m,s)+df["Forecast"]
limit_low = norm.ppf(0.05,m,s)+df["Forecast"]
df["Updated"] = df["Demand"].clip(lower=limit_low,upper=limit_high).astype(int)

df["Error"] = limit_high
df["Error Low"] = limit_low

m = df["Demand"].values.mean()
s = df["Demand"].values.std()

norm.cdf(df["Demand"].values, m, s).round(2)

limit_high = norm.ppf(0.95,m,s)
limit_low = norm.ppf(0.05,m,s)

###################################################

name = "chap10-winVSnorVSerr0-fig"
ax = df[["Demand","Forecast"]].plot(style=["-","--"],figsize=(8,3))
ax.set_ylabel("Volume")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 
plt.tight_layout()
plt.savefig(outputs+name+".pdf")

###################################################

name = "chap10-winVSnorVSerr1-fig"
ax = df[["Demand","Forecast"]].plot(style=["-","--"],figsize=(8,3))
df["Error Low"].plot(style=[":"],color=["green"],ax=ax,label='_nolegend_')
df["Error"].plot(style=[":"],color=["green"],ax=ax,label="Error limits")

ax.fill_between(df.index.values,df["Error Low"],df["Error"], facecolor='green', alpha=0.15)
ax.legend(loc="upper right")

ax.set_ylabel("Volume")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 
plt.tight_layout()
plt.savefig(outputs+name+".pdf")

###################################################

name = "chap10-winVSnorVSerr2-fig"
ax = df[["Demand"]].plot(style=["-","--"],figsize=(8,3))
limit_win_high = np.percentile(df["Demand"],99)
limit_win_low = np.percentile(df["Demand"],1)
plt.axhline(y=limit_high,color="firebrick",ls="--",label='Normalization',linewidth=2)
plt.axhline(y=limit_low,color="firebrick",ls="--",linewidth=2)
plt.axhline(y=limit_win_high,color="firebrick",ls="-",label='Winsorization',linewidth=2)
plt.axhline(y=limit_win_low,color="firebrick",ls="-",linewidth=2)
# ax.legend(loc="upper right")
# ax.legend(loc="upper center",bbox_to_anchor=(0.5, 1.2), ncol=3)
ax.legend(loc="best",bbox_to_anchor=(1, 1), ncol=1)
ax.set_ylabel("Volume")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 
plt.tight_layout()
plt.savefig(outputs+name+".pdf")

###################################################

name = "chap10-gaussian-fig"

mu = 0
variance = 1
sigma = np.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y = stats.norm.pdf(x, mu, sigma)
fig, ax = plt.subplots(figsize=(6,3))
ax.plot(x,y,color="k")
ax.fill_between(x, 0,y, facecolor='red', alpha=0.5,where=((x>2.33*sigma)|(x<-2.33*sigma)))

plt.text(-3.1, 0.04, '$-2.33\sigma$')
plt.text(2.4, 0.04, '$2.33\sigma$')
plt.xticks(np.arange(-3,4), ('$-3\sigma$', '$-2\sigma$', '$-1\sigma$', '0', '$1\sigma$',"$2\sigma$","3$\sigma$"))
ax.set_xlabel("Distance to the mean")
ax.set_ylabel("Density")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,
                labelbottom=False,labelleft=False) 

plt.tight_layout()
plt.savefig(outputs+name+".pdf")

#####################################