from ETS import * 
d = [14,10,6,2,18,8,4,1,16,9,5,3,18,11,4,2,17,9,5,1]
df = triple_exp_smooth_mul(d, slen=12, extra_periods=4, alpha=0.3, beta=0.2, phi=0.9, gamma=0.2)
kpi(df)
df.plot()
df.plot(subplots=True,figsize=(8,7))
df[["Level","Trend","Season"]].plot(figsize=(8,4),secondary_y=["Season"])
