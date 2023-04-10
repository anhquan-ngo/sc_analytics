import numpy as np
np.random.seed(0)
import pandas as pd
from support import datasets, import_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

# Idea \#1 -- Training Set 

df = import_data()
forest_features = {'n_jobs':-1, 'n_estimators':200,  'min_samples_split': 15, 'min_samples_leaf': 4, 'max_samples': 0.95, 'max_features': 0.3, 'max_depth': 8, 'bootstrap': True}
forest = RandomForestRegressor(**forest_features)
ETR_features = {'n_jobs':-1, 'n_estimators':200, 'min_samples_split': 14, 'min_samples_leaf': 2, 'max_samples': 0.9, 'max_features': 1.0, 'max_depth': 12, 'bootstrap': True}
ETR = ExtraTreesRegressor(**ETR_features)
models = [('Forest',forest), ('ETR',ETR)]

def model_mae(model, X, Y):
    Y_pred = model.predict(X)
    mae = np.mean(abs(Y - Y_pred))/np.mean(Y)
    return mae

n_months = range(6,50,2)
results = []
for x_len in n_months: # We loop through the different x_len
    X_train, Y_train, X_test, Y_test = datasets(df, x_len=x_len)  
    for name, model in models: # We loop through the models
        model.fit(X_train,Y_train)  
        mae_train = model_mae(model, X_train, Y_train)
        mae_test = model_mae(model, X_test, Y_test)  
        results.append([name+' Train',mae_train,x_len])          
        results.append([name+' Test',mae_test,x_len])
        
data = pd.DataFrame(results,columns=['Model','MAE%','Number of Months'])
data = data.set_index(['Number of Months','Model']).stack().unstack('Model')
data.index = data.index.droplevel(level=1)
data.index.name = 'Number of months'

data.plot(color=['orange']*2+['black']*2,style=['-','--']*2)
print(data.idxmin())

#################

#Idea \#2 -- Validation Set

from sklearn.model_selection import KFold
results = []
for x_len in n_months:      
    X_train, Y_train, X_test, Y_test = datasets(df, x_len=x_len)      
    for name, model in models:
        mae_kfold_train = []
        mae_kfold_val = []
        for train_index, val_index in KFold(n_splits=8).split(X_train):
            X_train_kfold, X_val_kfold = X_train[train_index], X_train[val_index]
            Y_train_kfold, Y_val_kfold = Y_train[train_index], Y_train[val_index]
            model.fit(X_train_kfold, Y_train_kfold) 
            mae_train = model_mae(model, X_train_kfold, Y_train_kfold) 
            mae_kfold_train.append(mae_train)                
            mae_val = model_mae(model, X_val_kfold, Y_val_kfold)
            mae_kfold_val.append(mae_val)      
        results.append([name+' Val',np.mean(mae_kfold_val),x_len])
        results.append([name+' Train',np.mean(mae_kfold_train),x_len])
        
        model.fit(X_train,Y_train)            
        mae_test = model_mae(model, X_test, Y_test)
        results.append([name+' Test',mae_test,x_len])

data = pd.DataFrame(results,columns=['Model','MAE%','Number of Months'])
data = data.set_index(['Number of Months','Model']).stack().unstack('Model')
data.index = data.index.droplevel(level=1)
data.index.name = 'Number of months'
data.plot(color=['orange']*3+['black']*3,style=['-','--',':']*2)  
print(data.idxmin())

#################

#Idea \#3 -- Holdout Dataset

def datasets_holdout(df, x_len=12, y_len=1, test_loops=12, holdout_loops=0):
    D = df.values
    rows, periods = D.shape
    
    # Training set creation
    train_loops = periods + 1 - x_len - y_len - test_loops 
    train = []
    for col in range(train_loops):
        train.append(D[:,col:col+x_len+y_len])
    train = np.vstack(train)
    X_train, Y_train = np.split(train,[-y_len],axis=1)
    
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
        X_test = D[:,-x_len:]     
        Y_test = np.full((X_test.shape[0],y_len),np.nan) #Dummy value
    
    # Formatting required for scikit-learn
    if y_len == 1: 
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()  
        Y_holdout = Y_holdout.ravel()
        
    return X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test


results = []
for x_len in n_months:   
    X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test = datasets_holdout(df, x_len=x_len, holdout_loops=12) 
    for name, model in models: 
        model.fit(X_train,Y_train)   
        mae_train = model_mae(model, X_train, Y_train)
        mae_holdout = model_mae(model, X_holdout, Y_holdout)
        mae_test = model_mae(model, X_test, Y_test)
        results.append([name+' Train',mae_train,x_len])
        results.append([name+' Test',mae_test,x_len])
        results.append([name+' Holdout',mae_holdout,x_len])
        
data = pd.DataFrame(results,columns=['Model','MAE%','Number of Months'])
data = data.set_index(['Number of Months','Model']).stack().unstack('Model')
data.index = data.index.droplevel(level=1)
data.index.name = 'Number of months'
data.plot(color=['orange']*3+['black']*3,style=['-','--',':']*3)  
print(data.idxmin())