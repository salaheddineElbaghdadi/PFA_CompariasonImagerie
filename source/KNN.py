import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors

inputDataFile = 'input/numericDataSet.csv'
k = 10

df = pd.read_csv(inputDataFile)

x = np.array(df.drop(['Group'], 1))
y = np.array(df['Group'])
df.reset_index()
#print(x)
#print(y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2)

clf = neighbors.KNeighborsClassifier(n_neighbors=k)
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
print(k, 'nn:')
print(accuracy)