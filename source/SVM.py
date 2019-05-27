import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


inputDataFile = 'input/numericDataSet.csv'
df = pd.read_csv(inputDataFile)
df.drop(['Unnamed: 0'], axis = 'columns', inplace = True)
X = np.array(df.drop(['Group'], 1))
y = np.array(df['Group'])
df.reset_index()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state=1, stratify=y)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

for kernel in ['linear', 'rbf','sigmoid']:
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train.ravel())
    y_pred = svm.predict(X_test)
    print(classification_report(y_test,y_pred))
    print('Accuracy of',kernel,'is :',svm.score(X_test,y_test))

