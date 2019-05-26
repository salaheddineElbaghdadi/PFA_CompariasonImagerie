import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
    
inputDataFile = '../input/numericDataSet.csv'

df = pd.read_csv(inputDataFile)

df.drop(['MR Delay'], axis = 'columns', inplace = True)
x = np.array(df.drop(['Group'], 1))
y = np.array(df['Group'])
df.reset_index()

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.2, random_state=1, stratify=y)


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#DNN
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(9, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(9, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=30)

print(model.evaluate(x_test, y_test))

#print(x_train)
#print(x_test)
print(y_test)