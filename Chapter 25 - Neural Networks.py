from sklearn.model_selection import RandomizedSearchCV  
from support import import_data, datasets, kpi_ML
df = import_data()
X_train, Y_train, X_test, Y_test = datasets(df)

activation = 'relu'
solver = 'adam'
early_stopping = True
n_iter_no_change = 50
validation_fraction = 0.1
tol = 0.0001
param_fixed = {'activation':activation, 'solver':solver, 'early_stopping':early_stopping, 
               'n_iter_no_change':n_iter_no_change, 'validation_fraction':validation_fraction, 'tol':tol}

from sklearn.neural_network import MLPRegressor
NN = MLPRegressor(hidden_layer_sizes=(20,20),**param_fixed, verbose=True).fit(X_train, Y_train)
Y_train_pred = NN.predict(X_train) 
Y_test_pred = NN.predict(X_test) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='NN')  

hidden_layer_sizes = [[neuron]*hidden_layer for neuron in range(10,60,10) for hidden_layer in range(2,7)]
alpha = [5,1,0.5,0.1,0.05,0.01,0.001]
learning_rate_init = [0.05,0.01,0.005,0.001,0.0005]
beta_1 = [0.85,0.875,0.9,0.95,0.975,0.99,0.995]
beta_2 = [0.99,0.995,0.999,0.9995,0.9999]
param_dist = {'hidden_layer_sizes':hidden_layer_sizes, 'alpha':alpha, 
              'learning_rate_init':learning_rate_init, 'beta_1':beta_1, 'beta_2':beta_2}

NN = MLPRegressor(**param_fixed)
NN_cv = RandomizedSearchCV(NN, param_dist, cv=10, verbose=2, n_jobs=-1, n_iter=200, scoring='neg_mean_absolute_error')
NN_cv.fit(X_train,Y_train)  

print('Tuned NN Parameters:', NN_cv.best_params_)
print()
Y_train_pred = NN_cv.predict(X_train) 
Y_test_pred = NN_cv.predict(X_test) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='NN optimized')