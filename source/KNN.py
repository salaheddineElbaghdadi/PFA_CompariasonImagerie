import pandas as pd
from sklearn import preprocessing, model_selection, neighbors
import numpy as np
import matplotlib.pyplot as plt


inputDataFile = 'input/numericDataSet.csv'
#k = 69

df = pd.read_csv(inputDataFile)
#print(df.columns.values)

df.drop(['Unnamed: 0'], axis = 'columns', inplace = True)
print(df.columns.values)



x = np.array(df.drop(['Group'], 1))
y = np.array(df['Group'])
df.reset_index()
#print(x)
#print(y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2)

#clf = neighbors.KNeighborsClassifier(n_neighbors=k)
#clf.fit(x_train, y_train)
#accuracy = clf.score(x_test, y_test)
#print(clf.predict(x_test))
#print(accuracy)

KNN = np.arange(1, 80)
train_accuracy = np.empty(len(KNN))
test_accuracy = np.empty(len(KNN))

# Loop over K values
for i, k in enumerate(KNN):
    pfa = neighbors.KNeighborsClassifier(n_neighbors=KNN[i])
    pfa.fit(x_train, y_train)

    # Compute traning and test data accuracy
    train_accuracy[i] = pfa.score(x_train, y_train)
    test_accuracy[i] = pfa.score(x_test, y_test)
    print('for K=:', i,' the accuracy is :', test_accuracy[i])

# Generate plot
plt.plot(KNN, test_accuracy, label='Testing dataset Accuracy')
plt.plot(KNN, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
fig1 = plt.gcf()
plt.show()
fig1.savefig('plt.png')
