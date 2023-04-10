import pandas as pd
import numpy as np
from support import import_data, kpi_ML, datasets
from xgboost.sklearn import XGBRegressor
from sklearn.cluster import KMeans 
from sklearn.model_selection import train_test_split

##############################################################################

XGBoost_features = {'subsample': 0.2, 'reg_lambda': 0.1, 'reg_alpha': 20, 'n_estimators': 1000, 'min_child_weight': 5, 'max_depth': 10, 'learning_rate': 0.005, 'colsample_bytree': 0.8, 'colsample_bynode': 1.0, 'colsample_bylevel': 0.9}
XGB = XGBRegressor(n_jobs=-1, **XGBoost_features)

##############################################################################

#Benchmark
df = import_data()
X_train, Y_train, X_test, Y_test = datasets(df, x_len=12, y_len=1, test_loops=12)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15)
fit_params = {'early_stopping_rounds':100,
            'eval_set':[(x_val, y_val)],
            'eval_metric':'mae',
            'verbose':False}
XGB = XGB.fit(x_train, y_train, **fit_params) 
print(XGB.best_iteration)

Y_train_pred = XGB.predict(X_train) 
Y_test_pred = XGB.predict(X_test) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='XGBoost Eval')  

##############################################################################

df = import_data()

# Exogenous factors
GDP = pd.read_excel('GDP.xlsx').set_index('Year')
dates = pd.to_datetime(df.columns,format='%Y-%m').year
X_GDP = [GDP.loc[date,'GDP'] for date in dates]

# Brand as dummy
df['Brand'] = df.index
df = pd.get_dummies(df, columns=['Brand'], prefix_sep='_')

# Segment as numerical value
luxury = ['Aston Martin','Bentley','Ferrari','Jaguar','Lamborghini','Lexus','Lotus','Maserati','McLaren','Porsche','Tesla']
premium = ['Audi','BMW','Cadillac','Infiniti','Land Rover','MINI','Mercedes-Benz']
low_cost = ['Dacia','Skoda']
df['Segment'] = 2
mask = df.index.isin(luxury)
df.loc[mask,'Segment'] = 4
mask = df.index.isin(premium)
df.loc[mask,'Segment'] = 3
mask = df.index.isin(low_cost)
df.loc[mask,'Segment'] = 1

def seasonal_factors(df,slen):
    s = pd.DataFrame(index=df.index)
    for i in range(slen):
        idx = [x for x in range(df.shape[1]) if x%slen==i] # Indices that correspond to this season
        s[i+1] = np.mean(df.iloc[:,idx],axis=1)       #+1 to put the right month number
    s = s.divide(s.mean(axis=1),axis=0).fillna(0)
    return s

def scaler(s):
    mean = s.mean(axis=1)
    maxi = s.max(axis=1)
    mini = s.min(axis=1)
    s = s.subtract(mean,axis=0)
    s = s.divide(maxi-mini,axis=0).fillna(0)
    return s

# Cluster as numerical value
s = seasonal_factors(df,slen=12)
s = scaler(s)
kmeans = KMeans(n_clusters=4, random_state=0).fit(s)
df['Group'] = kmeans.predict(s) 


def datasets_full(df, X_exo, x_len=12, y_len=1, test_loops=12, holdout_loops=0, cat_name=['_']):
    
    col_cat = [col for col in df.columns if any(name in col for name in cat_name)]
    D = df.drop(columns=col_cat).values # Historical demand
    C = df[col_cat].values # Categorical info
    rows, periods = D.shape
    X_exo = np.repeat(np.reshape(X_exo,[1,-1]),rows,axis=0)   
    X_months = np.repeat(np.reshape([int(col[-2:]) for col in df.columns if col not in col_cat],[1,-1]),rows,axis=0)   
    
    # Training set creation
    loops = periods + 1 - x_len - y_len
    train = []
    for col in range(loops):
        m = X_months[:,col+x_len].reshape(-1,1) #month
        exo = X_exo[:,col:col+x_len+y_len] #exogenous data
        exo = np.hstack([np.mean(exo,axis=1,keepdims=True),
                         np.mean(exo[:,-4:],axis=1,keepdims=True),
                         exo]) 
        d = D[:,col:col+x_len+y_len]
        d = np.hstack([np.mean(d[:,:-y_len],axis=1,keepdims=True),
                       np.median(d[:,:-y_len],axis=1,keepdims=True),
                       np.mean(d[:,-4-y_len:-y_len],axis=1,keepdims=True),
                       np.max(d[:,:-y_len],axis=1,keepdims=True),
                       np.min(d[:,:-y_len],axis=1,keepdims=True),
                       d])
        train.append(np.hstack([m, exo, d]))
    train = np.vstack(train)
    X_train, Y_train = np.split(train,[-y_len],axis=1)
    X_train = np.hstack((np.vstack([C]*loops),X_train))
    features = (col_cat
                +['Month']
                +['Exo Mean','Exo MA4']+[f'Exo M{-x_len+col}' for col in range(x_len+y_len)] 
                +['Demand Mean','Demand Median','Demand MA4','Demand Max','Demand Min']+[f'Demand M-{x_len-col}' for col in range(x_len)])

    # Holdout set creation
    if holdout_loops > 0:
        X_train, X_holdout = np.split(X_train,[-rows*holdout_loops],axis=0)
        Y_train, Y_holdout = np.split(Y_train,[-rows*holdout_loops],axis=0)
    else:
        X_holdout, Y_holdout = np.array([]), np.array([])
        
    # Test set creation
    if test_loops > 0:
        X_train, X_test = np.split(X_train,[-rows*test_loops],axis=0)
        Y_train, Y_test = np.split(Y_train,[-rows*test_loops],axis=0)
    else: # No test set: X_test is used to generate the future forecast
        exo = X_exo[:,-x_len-y_len:]
        d = D[:,-x_len:]
        X_test = np.hstack((C,
                            m[:,-1].reshape(-1,1),    
                            np.hstack([np.mean(exo,axis=1,keepdims=True),
                                       np.mean(exo[:,-4:],axis=1,keepdims=True),
                                       exo]),
                            np.hstack([np.mean(d,axis=1,keepdims=True),
                                       np.median(d,axis=1,keepdims=True),
                                       np.mean(d[:,-4:],axis=1,keepdims=True),
                                       np.max(d,axis=1,keepdims=True),
                                       np.min(d,axis=1,keepdims=True),
                                       d])))    
        Y_test = np.full((X_test.shape[0],y_len),np.nan) #Dummy value
    
    # Formatting required for scikit-learn
    if y_len == 1: 
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()  
        Y_holdout = Y_holdout.ravel()
        
    return X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test, features

X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test, features = datasets_full(df, X_GDP, x_len=12, y_len=1, test_loops=12, holdout_loops=0, cat_name=['_','Segment','Group'])
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15)

##############################################################################

XGB = XGB.fit(x_train, y_train, early_stopping_rounds=100, verbose=False, eval_set=[(x_val,y_val)], eval_metric='mae') 
Y_train_pred = XGB.predict(X_train) 
Y_test_pred = XGB.predict(X_test) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='XGBoost')   

imp = XGB.get_booster().get_score(importance_type='total_gain')
imp = pd.DataFrame.from_dict(imp,orient='index',columns=['Importance'])
imp.index = np.array(features)[imp.index.astype(str).str.replace('f','').astype(int)]
imp = (imp['Importance']/sum(imp.values)).sort_values(ascending=False)
imp.to_excel('Feature Importance.xlsx')

def model_kpi(model, X, Y):
    Y_pred = model.predict(X)
    mae = np.mean(abs(Y - Y_pred))/np.mean(Y)
    rmse = np.sqrt(np.mean((Y - Y_pred)**2))/np.mean(Y)
    return mae,rmse

results = []
limits = [0,0.00005,0.0001,0.00015,0.0002,0.00025,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.0011,0.002,0.004,0.008,0.01,0.02,0.04,0.06]
for limit in limits:
    mask = [feature in imp[imp > limit] for feature in features]
    XGB = XGB.fit(x_train[:,mask], y_train, early_stopping_rounds=100, verbose=False, eval_set=[(x_val[:,mask],y_val)], eval_metric='mae') 
    results.append(model_kpi(XGB,x_val[:,mask],y_val))
results = pd.DataFrame(data=results,columns=['MAE','RMSE'],index=limits)
results.plot(secondary_y='MAE',logx=True)

##############################################################################

limit = 0.0007
print(imp[imp > limit].index)
mask = [feature in imp[imp > limit] for feature in features]
XGB = XGB.fit(x_train[:,mask], y_train, early_stopping_rounds=100, verbose=False, eval_set=[(x_val[:,mask],y_val)], eval_metric='mae') 
Y_train_pred = XGB.predict(X_train[:,mask]) 
Y_test_pred = XGB.predict(X_test[:,mask]) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='XGBoost')   

##############################################################################

# Future forecast
# X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test, features = datasets_full(df, X_GDP, x_len=12, y_len=1, test_loops=0, holdout_loops=0, cat_name=['_','Segment','Group'])
# x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15)
# XGB = XGB.fit(x_train[:,mask], y_train, early_stopping_rounds=100, verbose=False, eval_set=[(x_val[:,mask],y_val)], eval_metric='mae') 
# forecast = pd.DataFrame(data=XGB.predict(X_test[:,mask]), index=df.index)
