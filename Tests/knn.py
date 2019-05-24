import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors

#df = pd.read_csv('oasis_cross-sectional.csv')
df = pd.read_csv('oasis_longitudinal.csv')
df.replace('', -99999, inplace = True)
df.replace('F', 0, inplace = True)
df.replace('M', 1, inplace = True)
df.replace('R', 1, inplace = True)
df.replace('L', 0, inplace = True)
df.replace('Demented', 1, inplace = True)
df.replace('Nondemented', 0, inplace = True)
df.replace('Converted', 1, inplace = True)
df.drop(['Subject ID', 'MRI ID'], axis = 1)
df = df.fillna(0)
#df = df.dropna()
#df.drop(['MRI ID'], axis = 1)
df.reset_index()

#np.any(np.isnan(mat))
#np.all(np.isfinite(mat))

print(df)

x = np.array(df.drop(['Subject ID', 'MRI ID','Group', 'M/F', 'Hand'], 1))
y = np.array(df['Group'])
df.reset_index()
#print(x)
#print(y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2)

clf = neighbors.KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(accuracy)