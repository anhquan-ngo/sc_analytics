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

name = "chap8-overfit-fig"
df = pd.read_csv(inputs+name+".csv")

ax = df[["Demand","Forecast"]].plot(style=['-', '--'],figsize=(8,3.5))
ax.set_xlabel("Period")
ax.set_ylabel("Volume")
#ax.set_title("Overfitting")
ax.text(5, 100, 'RMSE: 10.3%', size=12)
ax.text(35, 100, 'RMSE: 19.0%', size=12)
ax.text(5, 50, 'Training set', size=12)
ax.text(36, 50, 'Test set', size=12)
plt.axvline(x=20,color="black",ls=':')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tick_params(axis='both',which='both',bottom=True,top=False,left=False,
                labelbottom=True,labelleft=False) 

plt.tight_layout()
plt.savefig(outputs+name+".pdf")

########################################

import numpy as np
df["Error"] = df["Demand"] - df["Forecast"]
print(np.sqrt((df[:20]["Error"]**2).sum())/df[:20]["Demand"].sum())
print(np.sqrt((df[20:]["Error"]**2).sum())/df[20:]["Demand"].sum())
