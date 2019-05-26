import pandas as pd
from sklearn import model_selection, neighbors
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

maxDesK = 60
inputDataFile = '../input/numericDataSet.csv'


df = pd.read_csv(inputDataFile)
df.drop(['Unnamed: 0'], axis = 'columns', inplace = True)



x = np.array(df.drop(['Group'], 1))
y = np.array(df['Group'])
df.reset_index()
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2, random_state=1, stratify=y)


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



KNN = np.arange(1, maxDesK)
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


for i in range(maxDesK-1):
    if test_accuracy[i]==max(test_accuracy):
        print('The MAX acurracy is :',max(test_accuracy),'For k=',i)

# Generate plot
plt.plot(KNN, test_accuracy, label='Testing dataset Accuracy')
plt.plot(KNN, train_accuracy, label='Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
fig1 = plt.gcf()
plt.show()
fig1.savefig('AccuracyAfterScalling.png')

