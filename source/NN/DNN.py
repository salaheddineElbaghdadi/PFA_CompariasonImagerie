import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def neuralNetworkAccuracyTest(hl1, hl2):
    minInLayer = 2
    maxAccuracy = 0
    bestTopology = (0,0)
    nnTopology = []
    accuracy = []

    for i in range(minInLayer, hl1):
        for j in range(minInLayer, hl2):
            #DNN
            model = tf.keras.models.Sequential()
            #model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(i, input_shape=(10,), activation=tf.nn.relu))
            model.add(tf.keras.layers.Dense(j, activation=tf.nn.relu))
            model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=30)

            accuracy.append(model.evaluate(x_test, y_test)[1])
            nnTopology.append((i,j))
    
    for i in range(0, len(accuracy)):
        print('nn topoloty:', nnTopology[i], '  accuracy:', accuracy[i])
        if accuracy[i] > maxAccuracy:
            maxAccuracy = accuracy[i]
            bestTopology = nnTopology[i]
    
    print('Best topoloty: ', bestTopology, '  with accuracy: ', maxAccuracy)
    

    
inputDataFile = '../input/numericDataSet.csv'

# reading data from csv file
df = pd.read_csv(inputDataFile)

# dividing data to input and output
x = np.array(df.drop(['Group'], 1))
y = np.array(df['Group'])
df.reset_index()

# dividing data to training set and testing set
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2, random_state=1, stratify=y)

# normalizing data
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

'''
#DNN
model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(2, input_shape=(10,), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30)

evaluation = model.evaluate(x_test, y_test)
print('Evaluation: ', evaluation)
'''

neuralNetworkAccuracyTest(10, 10)

#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

#print(model.predict(x_test))