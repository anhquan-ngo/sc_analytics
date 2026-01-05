import numpy as np
import pandas as pd
from scipy.stats import norm

# populate a demand array that follows a normal distribution.
time = 200
d_mu = 100
d_std = 25
d = np.maximum(np.random.normal(d_mu,d_std,time).round(0).astype(int),0)

# define the remaining input
L, R, alpha = 4, 1, 0.95
z = norm.ppf(alpha)
x_std = np.sqrt(L+R)*d_std
Ss = np.round(x_std*z).astype(int)
Cs = 1/2 * d_mu * R
Is = d_mu * L
S = Ss + 2*Cs + Is

# 2 arrays to keep track of the on-hand and in-transit inventory per period
hand = np.zeros(time,dtype=int)
transit = np.zeros((time,L+1),dtype=int)

# initialize these arrays for period 0
hand[0] = S - d[0]
transit[1,-1] = d[0]

# stock_out_period will contain a Boolean that flags if there is a shortage during a period.
stockout_period = np.full(time,False,dtype=bool)
stockout_cycle = []

# 1. Check whether we received an order at the beginning of the period (transit[ t-1,0] > 0). If so, we need to compute the cycle service level by checking if there was a stock-out last period. Remember, we define the cycle service level as the probability that there is no stock-out just before an order is received.
# 2. Update the on-hand inventory by subtracting the demand d[t] and adding the received inventory transit[t-1,0] 
# 3. Indicate in stockout_period[t] whether we had a shortage.
# 4. Update the net inventory position net[t]. Remember that it is the total intransit inventory transit[t].sum() plus the on-hand inventory hand[t]
# Note that you can exclude backorders (i. e., excess demand will be lost) by uncommenting the line hand[t] = max(0,hand[t]).
# 5. Update the in-transit array by offsetting the values of the previous timestep by 1: transit[t,:-1] = transit[t-1,1:]. This represents the fact that the orders move through the supply pipeline.
# 6. If we are at the review period (t%R==0), we make an order based on the current net inventory position net[t] and the up-to level S. The order is then stored at the extreme of the in-transit array transit[t,L].

for t in range(1,time):
    if transit[t-1,0]>0:
        stockout_cycle.append(stockout_period[t-1])
    hand[t] = hand[t-1] - d[t] + transit[t-1,0]
    stockout_period[t] = hand[t] < 0
    #hand[t] = max(0,hand[t]) #Uncomment if excess demand result in lost sales rather than backorders
    transit[t,:-1] = transit[t-1,1:]
    if 0==t%R:
        net = hand[t] + transit[t].sum()
        transit[t,L] = S - net


df = pd.DataFrame(data= {'Demand':d,'On-hand':hand,'In-transit':
list(transit)})
df = df.iloc[R+L:,:] #Remove initialization periods
print(df)
df['On-hand'].plot(title='Inventory Policy (%d,%d)' %(R,S), ylim
=(0,S), legend=True)
print('Alpha:',alpha*100)
SL_alpha = 1-sum(stockout_cycle)/len(stockout_cycle)
print('Cycle Service Level:',round(SL_alpha*100,1))
SL_period = 1-sum(stockout_period)/time
print('Period Service Level:',round(SL_period*100,1))