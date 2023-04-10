import pandas as pd
import numpy as np
np.random.seed(0)
from support import import_data, datasets_holdout, datasets, kpi_ML

df = import_data()
 
###########################################################

X_train, Y_train, X_test, Y_test = datasets(df, x_len=12, y_len=1, test_loops=12)

from xgboost.sklearn import XGBRegressor
XGB = XGBRegressor(n_jobs=-1, max_depth=10, n_estimators=100, learning_rate=0.2)  
XGB = XGB.fit(X_train, Y_train) 
   
import xgboost as xgb
xgb.plot_importance(XGB, importance_type='total_gain', show_values=False)

###########################################################

from sklearn.multioutput import MultiOutputRegressor
X_train, Y_train, X_test, Y_test = datasets(df, x_len=12, y_len=6, test_loops=12)
XGB = XGBRegressor(n_jobs=1, max_depth=10, n_estimators=100, learning_rate=0.2)  
multi = MultiOutputRegressor(XGB, n_jobs=-1)    
multi.fit(X_train,Y_train) 
 
# Future Forecast
X_train, Y_train, X_test, Y_test = datasets(df, x_len=12, y_len=6, test_loops=0)
XGB = XGBRegressor(n_jobs=1, max_depth=10, n_estimators=100, learning_rate=0.2)  
multi = MultiOutputRegressor(XGB, n_jobs=-1)    
multi.fit(X_train,Y_train) 
forecast = pd.DataFrame(data=multi.predict(X_test), index=df.index)

###########################################################

X_train, Y_train, X_test, Y_test = datasets(df, x_len=12, y_len=1, test_loops=12)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15)

XGB = XGBRegressor(n_jobs=-1, max_depth=10, n_estimators=2000, learning_rate=0.01)  
XGB = XGB.fit(x_train, y_train, early_stopping_rounds=100, verbose=True, eval_set=[(x_val, y_val)], eval_metric='mae') 

XGB = XGBRegressor(n_jobs=-1, max_depth=10, n_estimators=2000, learning_rate=0.01)  
XGB = XGB.fit(x_train, y_train, early_stopping_rounds=100, verbose=False, eval_set=[(x_train, y_train), (x_val, y_val)], eval_metric='mae', 
callbacks = [xgb.callback.print_evaluation(period=50)])
print(f'Best iteration: {XGB.get_booster().best_iteration}')
print(f'Best score: {XGB.get_booster().best_score}')

# multi periods and with fit_params
# from sklearn.model_selection import train_test_split
# X_train, Y_train, X_test, Y_test = datasets(df, x_len=12, y_len=6, test_loops=12)
# x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15)
# fit_params = {'early_stopping_rounds':25,
#             'eval_set':[(x_val, y_val)],
#             'eval_metric':'mae',
#             'verbose':False}
# XGB = XGBRegressor(n_jobs=1, max_depth=10, n_estimators=100, learning_rate=0.2)  
# multi = MultiOutputRegressor(XGB, n_jobs=-1)    
# multi.fit(X_train,Y_train,**fit_params) 

###########################################################

# Fit with holdout dataset
# X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test = datasets_holdout(df, x_len=12, y_len=1, test_loops=12, holdout_loops=12)

# XGB = XGBRegressor(n_jobs=-1, max_depth=10, n_estimators=2000, learning_rate=0.01)  
# XGB = XGB.fit(X_train, Y_train, early_stopping_rounds=100, verbose=False, eval_set=[(X_train, Y_train), (X_holdout, Y_holdout)], eval_metric='mae', 
# callbacks = [xgb.callback.print_evaluation(period=50)])
# print(f'Best iteration: {XGB.get_booster().best_iteration}')
# print(f'Best score: {XGB.get_booster().best_score}')

############################################################

X_train, Y_train, X_test, Y_test = datasets(df, x_len=12, y_len=1, test_loops=12)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.15)

params = {'max_depth': [5,6,7,8,10,11],
        'learning_rate': [0.005,0.01,0.025,0.05,0.1,0.15],
        'colsample_bynode' : [0.5,0.6,0.7,0.8,0.9,1.0],#max_features
        'colsample_bylevel': [0.8,0.9,1.0],
        'colsample_bytree': [0.6,0.7,0.8,0.9,1.0],
        'subsample': [0.1,0.2,0.3,0.4,0.5,0.6,0.7],#max_samples
        'min_child_weight': [5,10,15,20,25],#min_samples_leaf
        'reg_alpha': [1,5,10,20,50],
        'reg_lambda': [0.01,0.05,0.1,0.5,1],
        'n_estimators':[1000]}

fit_params = {'early_stopping_rounds':25,
            'eval_set':[(x_val, y_val)],
            'eval_metric':'mae',
            'verbose':False}

from sklearn.model_selection import RandomizedSearchCV

XGB = XGBRegressor(n_jobs=1)  
XGB_cv = RandomizedSearchCV(XGB, params, cv=5, n_jobs=-1, verbose=1, n_iter=1000, scoring='neg_mean_absolute_error')  
XGB_cv.fit(x_train, y_train,**fit_params)  
print('Tuned XGBoost Parameters:',XGB_cv.best_params_)

best_params = XGB_cv.best_params_
XGB = XGBRegressor(n_jobs=-1, **best_params)  
XGB = XGB.fit(x_train, y_train, **fit_params) 
print(f'Best iteration: {XGB.get_booster().best_iteration}')
print(f'Best score: {XGB.get_booster().best_score}')
Y_train_pred = XGB.predict(X_train) 
Y_test_pred = XGB.predict(X_test) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='XGBoost')  

############################################################

# Same with holdout dataset
# X_train, Y_train, X_holdout, Y_holdout, X_test, Y_test = datasets_holdout(df, x_len=12, y_len=1, test_loops=12, holdout_loops=12)

# fit_params = {'early_stopping_rounds':25,
#             'eval_set':[(X_holdout, Y_holdout)],
#             'eval_metric':'mae',
#             'verbose':False}

# XGB = XGBRegressor(n_jobs=1)  
# XGB_cv = RandomizedSearchCV(XGB, params, cv=5, n_jobs=-1, verbose=1, n_iter=1000, scoring='neg_mean_absolute_error')  
# XGB_cv.fit(X_train, Y_train,**fit_params)  
# print('Tuned XGBoost Parameters:',XGB_cv.best_params_)

# best_params = XGB_cv.best_params_

# XGB = XGBRegressor(n_jobs=-1, **best_params)  
# XGB = XGB.fit(X_train, Y_train, **fit_params) 
# print(f'Best iteration: {XGB.get_booster().best_iteration}')
# print(f'Best score: {XGB.get_booster().best_score}')
# Y_train_pred = XGB.predict(X_train) 
# Y_test_pred = XGB.predict(X_test) 
# kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='XGBoost')  
