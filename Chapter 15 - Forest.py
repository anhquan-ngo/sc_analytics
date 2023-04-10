import numpy as np
np.random.seed(0)
from support import import_data, datasets, kpi_ML
df = import_data()
X_train, Y_train, X_test, Y_test = datasets(df)

###############################################################

from sklearn.ensemble import RandomForestRegressor  
forest = RandomForestRegressor(bootstrap=True, max_samples=0.95, max_features=11, min_samples_leaf=18, max_depth=7)  
forest.fit(X_train,Y_train)  
 
Y_train_pred = forest.predict(X_train) 
Y_test_pred = forest.predict(X_test) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Forest')

##############################################################

from sklearn.model_selection import RandomizedSearchCV 

max_depth = list(range(5,11)) + [None]
min_samples_split = range(5,20)
min_samples_leaf = range(2,15)
max_features = range(3,8)
bootstrap = [True] #We force bootstrap
max_samples = [.7,.8,.9,.95,1]

param_dist = {'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'max_features': max_features,
              'bootstrap': bootstrap,
              'max_samples': max_samples}

forest = RandomForestRegressor(n_jobs=1, n_estimators=30)
forest_cv = RandomizedSearchCV(forest, param_dist, cv=6, n_jobs=-1, verbose=2, n_iter=400, scoring='neg_mean_absolute_error')
forest_cv.fit(X_train,Y_train)

print('Tuned Forest Parameters:', forest_cv.best_params_)
print()
Y_train_pred = forest_cv.predict(X_train) 
Y_test_pred = forest_cv.predict(X_test) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Forest optimized')

##############################################################

forest = RandomForestRegressor(n_estimators=200, n_jobs=-1, **forest_cv.best_params_).fit(X_train, Y_train)
Y_train_pred = forest.predict(X_train) 
Y_test_pred = forest.predict(X_test) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Forestx200')