from support import import_data, datasets, kpi_ML

df = import_data()
X_train, Y_train, X_test, Y_test = datasets(df)

from sklearn.tree import DecisionTreeRegressor  
# Instantiate a Decision Tree Regressor  
tree = DecisionTreeRegressor(max_depth=5, min_samples_split=15, min_samples_leaf=5)   
# Fit the tree to the training data  
tree.fit(X_train,Y_train)  


Y_train_pred = tree.predict(X_train) 
Y_test_pred = tree.predict(X_test) 
kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name='Tree')


print()
import time
for criterion in ['mse','mae']:
    start_time = time.time()
    tree = DecisionTreeRegressor(max_depth=5, min_samples_split=15, min_samples_leaf=5, criterion=criterion)   
    tree.fit(X_train,Y_train)
    Y_train_pred = tree.predict(X_train) 
    Y_test_pred = tree.predict(X_test) 
    kpi_ML(Y_train, Y_train_pred, Y_test, Y_test_pred, name=f'Tree {criterion}')
    print('{:0.2f} seconds'.format(time.time() - start_time))
    print()
    
    
from sklearn.tree import plot_tree   
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15,6), dpi=300)
ax = fig.gca()
plot_tree(tree, fontsize=3, feature_names=[f'M{x-12}' for x in range(12)], rounded=True, filled=True, ax=ax)
fig.savefig('Regression Tree.PNG')

