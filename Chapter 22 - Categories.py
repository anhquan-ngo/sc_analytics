import pandas as pd
import numpy as np

from support import import_data, datasets, kpi_ML 
df = import_data()

######################

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
print(df)

######################

df['Brand'] = df.index
df['Brand'] = df['Brand'].astype('category').cat.codes
print(df)

# df['Brand'] = df.index.astype('category').codes  #Only works because it is the index

######################

df['Brand'] = df.index
df = pd.get_dummies(df, columns=['Brand'], prefix_sep='_')
print(df)

######################

def datasets_cat(df, x_len=12, y_len=1, test_loops=12, cat_name='_'):

    col_cat = [col for col in df.columns if cat_name in col]
    D = df.drop(columns=col_cat).values # Historical demand
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

######################

from sklearn.ensemble import ExtraTreesRegressor 
ETR_features = {'n_jobs':-1, 'n_estimators':200, 'min_samples_split': 14, 'min_samples_leaf': 2, 'max_samples': 0.9, 'max_features': 1.0, 'max_depth': 12, 'bootstrap': True}

df = import_data()
X_train, Y_train, X_test, Y_test = datasets(df, x_len=12, y_len=1, test_loops=12)
ETR = ExtraTreesRegressor(**ETR_features).fit(X_train,Y_train)
Y_train_pred = ETR.predict(X_train)
Y_test_pred = ETR.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='ETR')

df = import_data()
df['Segment'] = 2
mask = df.index.isin(luxury)
df.loc[mask,'Segment'] = 4
mask = df.index.isin(premium)
df.loc[mask,'Segment'] = 3
mask = df.index.isin(low_cost)
df.loc[mask,'Segment'] = 1
X_train, Y_train, X_test, Y_test = datasets_cat(df, x_len=12, y_len=1, test_loops=12, cat_name='Segment')
ETR = ExtraTreesRegressor(**ETR_features).fit(X_train,Y_train)
Y_train_pred = ETR.predict(X_train)
Y_test_pred = ETR.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='ETR (segment)')

df = import_data()
df['Brand'] = df.index
df = pd.get_dummies(df, columns=['Brand'], prefix_sep='_')
X_train, Y_train, X_test, Y_test = datasets_cat(df, x_len=12, y_len=1, test_loops=12, cat_name='_')
ETR = ExtraTreesRegressor(**ETR_features).fit(X_train,Y_train)
Y_train_pred = ETR.predict(X_train)
Y_test_pred = ETR.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='ETR (brands)')

# Future forecast
X_train, Y_train, X_test, Y_test = datasets_cat(df, x_len=12, y_len=1, test_loops=0, cat_name='_')
ETR = ExtraTreesRegressor(**ETR_features).fit(X_train,Y_train)
forecast = pd.DataFrame(data=ETR.predict(X_test), index=df.index)