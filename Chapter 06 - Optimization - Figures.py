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

d = [28,19,18,13,19,16,19,18,13,16,16,11,18,15,13,15,13,11,13,10,12]   
df = exp_smooth_opti(d)
df.index.name = 'Period'
df[['Demand','Forecast']].plot(figsize=(8,3),title='Best model found',ylim=(0,30),style=['-','--'])  

plt.tight_layout()
plt.savefig(outputs+'chap6-optimization-example-fig'+'.pdf')
