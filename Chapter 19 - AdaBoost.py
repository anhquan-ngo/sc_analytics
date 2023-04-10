import pandas as pd
import numpy as np
from support import import_data, datasets, kpi_ML
df = import_data()
X_train, Y_train, X_test, Y_test = datasets(df)

#####################################################

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8), n_estimators=100, learning_rate=0.25, loss='square')
ada = ada.fit(X_train,Y_train)  

#####################################################

Y_train_pred = ada.predict(X_train)
Y_test_pred = ada.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='AdaBoost')

#####################################################

def model_mae(model, X, Y):
    Y_pred = model.predict(X)
    mae = np.mean(np.abs(Y - Y_pred))/np.mean(Y)
    return mae

from sklearn.model_selection import RandomizedSearchCV 
X_train, Y_train, X_test, Y_test = datasets(df, test_loops=12)

#n_estimators = [100]
learning_rate = [0.005,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5]
loss = ['square','exponential','linear']
param_dist = {#'n_estimators': n_estimators,
              'learning_rate': learning_rate,
              'loss':loss}

####################################################

results = []
for max_depth in range(2,18,2):
    ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth))
    ada_cv = RandomizedSearchCV(ada, param_dist, n_jobs=-1, cv=6, n_iter=20, scoring='neg_mean_absolute_error')
    ada_cv.fit(X_train,Y_train) 
    print('Tuned AdaBoost Parameters:',ada_cv.best_params_)
    print('Result:',ada_cv.best_score_)    
    results.append([ada_cv.best_score_,ada_cv.best_params_,max_depth])

results = pd.DataFrame(data=results, columns=['Score','Best Params','Max Depth'])
optimal = results['Score'].idxmax()
print(results.iloc[optimal])

ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8),n_estimators=100,learning_rate=0.005,loss='exponential')
ada = ada.fit(X_train,Y_train)  
Y_train_pred = ada.predict(X_train)
Y_test_pred = ada.predict(X_test)
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='AdaBoost Optimized')

###################################################

from sklearn.multioutput import MultiOutputRegressor
X_train, Y_train, X_test, Y_test = datasets(df, y_len=6)
multi = MultiOutputRegressor(ada, n_jobs=-1)    
multi.fit(X_train,Y_train)    