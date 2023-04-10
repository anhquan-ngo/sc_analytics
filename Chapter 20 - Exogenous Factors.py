import pandas as pd
import numpy as np
from support import import_data

##############################################################

df = import_data()
GDP = pd.read_excel('GDP.xlsx').set_index('Year')
dates = pd.to_datetime(df.columns,format='%Y-%m').year
X_GDP = [GDP.loc[date,'GDP'] for date in dates]

##############################################################

strings = ['2020-10-31','2020-11-30','2020-12-31']
dates = pd.to_datetime(strings,format='%Y-%m-%d')
print(dates.year)
print(dates.month)
print(dates.day)

##############################################################

def datasets_exo(df, X_exo, x_len=12, y_len=1, test_loops=12):
    D = df.values
    rows, periods = D.shape
    X_exo = np.repeat(np.reshape(X_exo,[1,-1]),rows,axis=0)   
    X_months = np.repeat(np.reshape([int(col[-2:]) for col in df.columns],[1,-1]),rows,axis=0)   
 
    # Training set creation
    loops = periods + 1 - x_len - y_len
    train = []
    for col in range(loops):
        m = X_months[:,col+x_len].reshape(-1,1) #month
        exo = X_exo[:,col:col+x_len+y_len] #exogenous data
        d = D[:,col:col+x_len+y_len]
        train.append(np.hstack([m, exo, d]))
    train = np.vstack(train)
    X_train, Y_train = np.split(train,[-y_len],axis=1)

    # Test set creation
    if test_loops > 0:
        X_train, X_test = np.split(X_train,[-rows*test_loops],axis=0)
        Y_train, Y_test = np.split(Y_train,[-rows*test_loops],axis=0)
    else: # No test set: X_test is used to generate the future forecast
        X_test = np.hstack([m[:,-1].reshape(-1,1),X_exo[:,-x_len-y_len:],D[:,-x_len:]])    
        Y_test = np.full((X_test.shape[0],y_len),np.nan) #Dummy value
    
    # Formatting required for scikit-learn
    if y_len == 1: 
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()  
        
    return X_train, Y_train, X_test, Y_test

#####################################################

from sklearn.ensemble import ExtraTreesRegressor  
X_train, Y_train, X_test, Y_test = datasets_exo(df,X_GDP)
params={'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 10, 'max_depth': 11, 'bootstrap': False}
ETR = ExtraTreesRegressor(**params)
ETR.fit(X_train,Y_train)  

Y_train_pred = ETR.predict(X_train)
MAE_train = np.mean(abs(Y_train - Y_train_pred))/np.mean(Y_train)
print('ETR on training set MAE%:',round(MAE_train*100,1))

Y_test_pred = ETR.predict(X_test)
MAE_test = np.mean(abs(Y_test - Y_test_pred))/np.mean(Y_test)
print('ETR on test set MAE%:',round(MAE_test*100,1))

#####################################################
#future forecast
X_train, Y_train, X_test, Y_test = datasets_exo(df, X_GDP, test_loops=0)
ETR = ExtraTreesRegressor(**params)
ETR.fit(X_train,Y_train)  
forecast = pd.DataFrame(data=ETR.predict(X_test), index=df.index)