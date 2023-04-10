import pandas as pd
from support import import_data, datasets
df = import_data()
X_train, Y_train, X_test, Y_test = datasets(df)

from sklearn.ensemble import RandomForestRegressor  
forest = RandomForestRegressor(n_estimators=200, min_samples_split=12,  
                               min_samples_leaf=8, max_features=4, max_depth=9, bootstrap=False)
forest.fit(X_train,Y_train)  

######################################################################################

cols = X_train.shape[1]
features = [f'M-{cols-col}' for col in range(cols)]  
data = forest.feature_importances_.reshape(-1,1)
imp = pd.DataFrame(data=data, index=features, columns=['Forest'])
imp.plot(kind='bar')