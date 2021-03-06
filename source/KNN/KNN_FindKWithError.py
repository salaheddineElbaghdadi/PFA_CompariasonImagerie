import pandas as pd
from sklearn import preprocessing, model_selection, neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


import numpy as np
import matplotlib.pyplot as plt

maxDesK = 60
inputDataFile = '../input/numericDataSet.csv'

# reading data from csv file
df = pd.read_csv(inputDataFile)

# dividing data to input set and output set
x = np.array(df.drop(['Group'], 1))
y = np.array(df['Group'])
df.reset_index()

# dividing data to training set and testing set
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2, random_state=1, stratify=y)

# nomalizing data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


error = []

# Testing KNN for different k values
for i in range(1, maxDesK):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

# Finding min value of error
for i in range(maxDesK-1):
    if error[i]==min(error):
        print('The MIN Error is :',min(error),'For k=',i)

# Generate plot
plt.figure(figsize=(12, 6))
plt.plot(range(1, maxDesK), error, color='blue', marker='o',
         markerfacecolor='gold', markersize=10)
plt.title('FINDING K BASED ON MINIMUM ERROR')
plt.xlabel('n_neighbors')
plt.ylabel('Mean Error')
fig = plt.gcf()
plt.show()
fig.savefig('FindingKWithErrorCalculation.png')
