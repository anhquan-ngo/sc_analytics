import numpy as np
np.random.seed(0)
from support import import_data, datasets, kpi_ML
df = import_data()
X_train, Y_train, X_test, Y_test = datasets(df)

#####################################################

from sklearn.ensemble import ExtraTreesRegressor    
ETR = ExtraTreesRegressor(n_jobs=-1, n_estimators=200, min_samples_split=15, min_samples_leaf=4, max_samples=0.95, max_features=4, max_depth=8, bootstrap=True)
ETR.fit(X_train,Y_train)  

#####################################################
 
Y_train_pred = ETR.predict(X_train) 
Y_test_pred = ETR.predict(X_test) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='ETR') 

#####################################################

from sklearn.model_selection import RandomizedSearchCV  
 
max_depth = list(range(6,13)) + [None]
min_samples_split = range(7,16)
min_samples_leaf = range(2,13)
max_features = range(5,13)
bootstrap = [True] #We force bootstrap
max_samples = [.7,.8,.9,.95,1]

param_dist = {'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'max_features': max_features,
              'bootstrap': bootstrap,
              'max_samples': max_samples}

ETR = ExtraTreesRegressor(n_jobs=1, n_estimators=30)
ETR_cv = RandomizedSearchCV(ETR, param_dist, cv=5, verbose=2, n_jobs=-1, n_iter=400, scoring='neg_mean_absolute_error')
ETR_cv.fit(X_train,Y_train)  

print('Tuned ETR Parameters:', ETR_cv.best_params_)
print()
Y_train_pred = ETR_cv.predict(X_train) 
Y_test_pred = ETR_cv.predict(X_test) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='ETR optimized')

##############################################################

ETR = ExtraTreesRegressor(n_estimators=200, n_jobs=-1, **ETR_cv.best_params_).fit(X_train, Y_train)
Y_train_pred = ETR.predict(X_train) 
Y_test_pred = ETR.predict(X_test) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='ETRx200')
