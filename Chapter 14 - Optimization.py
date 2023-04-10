import pandas as pd
import numpy as np
np.random.seed(0)
from support import import_data, datasets, kpi_ML
df = import_data()
X_train, Y_train, X_test, Y_test = datasets(df)
from sklearn.tree import DecisionTreeRegressor

###############################################################

max_depth = list(range(5,11)) + [None]
min_samples_split = range(5,20)
min_samples_leaf = range(2,20)
param_dist = {'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}


from sklearn.model_selection import RandomizedSearchCV

tree = DecisionTreeRegressor()
tree_cv = RandomizedSearchCV(tree, param_dist, n_jobs=-1, cv=10, verbose=1, n_iter=100, scoring='neg_mean_absolute_error')
tree_cv.fit(X_train,Y_train)

print('Tuned Regression Tree Parameters:',tree_cv.best_params_)
print()


Y_train_pred = tree_cv.predict(X_train) 
Y_test_pred = tree_cv.predict(X_test) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Tree')


df = pd.DataFrame(tree_cv.cv_results_)
df_params = pd.DataFrame(df['params'].values.tolist())
df = pd.concat([df_params,df],axis=1)
df.to_excel('Results.xlsx')