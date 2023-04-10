import pandas as pd
import numpy as np
from support import import_data, datasets
df = import_data()
X_train, Y_train, X_test, Y_test = datasets(df)

from sklearn.linear_model import LinearRegression
reg = LinearRegression() # Create a linear regression object
reg = reg.fit(X_train,Y_train) # Fit it to the training data
# Create two predictions for the training and test sets
Y_train_pred = reg.predict(X_train)
Y_test_pred = reg.predict(X_test)
    
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
    
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Regression')


# Future Forecast
X_train, Y_train, X_test, Y_test = datasets(df, x_len=12, y_len=1, test_loops=0)
reg = LinearRegression()
reg = reg.fit(X_train,Y_train) 
forecast = pd.DataFrame(data=reg.predict(X_test), index=df.index)
