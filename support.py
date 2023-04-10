import pandas as pd
import numpy as np


def import_data():
    data = pd.read_csv('norway_new_car_sales_by_make.csv')
    data['Period'] = data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2)
    df = pd.pivot_table(data=data,values='Quantity',index='Make',columns='Period',aggfunc='sum',fill_value=0)
    return df


def datasets(df, x_len=12, y_len=1, test_loops=12):
    D = df.values
    rows, periods = D.shape
    
    # Training set creation
    loops = periods + 1 - x_len - y_len
    train = []
    for col in range(loops):
        train.append(D[:,col:col+x_len+y_len])
    train = np.vstack(train)
    X_train, Y_train = np.split(train,[-y_len],axis=1)

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
        
    return X_train, Y_train, X_test, Y_test

    
def kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name=''):
    df = pd.DataFrame(columns = ['MAE','RMSE','Bias'],index=['Train','Test'])
    df.index.name = name
    df.loc['Train','MAE'] = 100*np.mean(abs(Y_train - Y_train_pred))/np.mean(Y_train)
    df.loc['Train','RMSE'] = 100*np.sqrt(np.mean((Y_train - Y_train_pred)**2))/np.mean(Y_train)
    df.loc['Train','Bias'] = 100*np.mean((Y_train - Y_train_pred))/np.mean(Y_train)
    df.loc['Test','MAE'] = 100*np.mean(abs(Y_test - Y_test_pred))/np.mean(Y_test) 
    df.loc['Test','RMSE'] = 100*np.sqrt(np.mean((Y_test - Y_test_pred)**2))/np.mean(Y_test) 
    df.loc['Test','Bias'] = 100*np.mean((Y_test - Y_test_pred))/np.mean(Y_test) 
    df = df.astype(float).round(1) #Round number for display
    print(df)
    

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


def datasets_cat(df, x_len=12, y_len=1, test_loops=12, cat_name='_'):

    col_cat = [col for col in df.columns if cat_name in col]
    D = df.drop(col_cat,axis=1).values # Historical demand
    C = df[col_cat].values # Categorical info
    rows, periods = D.shape
    
    # Training set creation
    loops = periods + 1 - x_len - y_len
    train = []
    for col in range(loops):
        train.append(D[:,col:col+x_len+y_len])
    train = np.vstack(train)
    X_train, Y_train = np.split(train,[-y_len],axis=1)
    X_train = np.hstack((np.vstack([C]*loops),X_train))
    
    # Test set creation
    if test_loops > 0:
        X_train, X_test = np.split(X_train,[-rows*test_loops],axis=0)
        Y_train, Y_test = np.split(Y_train,[-rows*test_loops],axis=0)
    else: # No test set: X_test is used to generate the future forecast
        X_test = np.hstack((C,D[:,-x_len:]))    
        Y_test = np.full((X_test.shape[0],y_len),np.nan) #Dummy value
    
    # Formatting required for scikit-learn
    if y_len == 1: 
        Y_train = Y_train.ravel()
        Y_test = Y_test.ravel()  
        
    return X_train, Y_train, X_test, Y_test


def datasets_exo(df, X_exo, x_len=12, y_len=1, test_loops=12):
    D = df.values
    rows, periods = D.shape
    X_exo = np.repeat(np.reshape(X_exo,[1,-1]),rows,axis=0)   
    X_months = np.repeat(np.reshape([int(col[-2:]) for col in df.columns],[1,-1]),rows,axis=0)   
 
    # Training set creation
    loops = periods + 1 - x_len - y_len
    train = []
    for col in range(loops):
        d = D[:,col:col+x_len+y_len]
        exo = X_exo[:,col:col+x_len+y_len]
        m = X_months[:,col+x_len].reshape(-1,1) # 
        train.append(np.hstack([m,exo,d]))
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

