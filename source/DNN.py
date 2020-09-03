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

import tensorflow as tf
from sklearn import metrics
X_FEATURE = 'x'  # Name of the input feature.
feature_columns = [
      tf.feature_column.numeric_column(
          X_FEATURE, shape=np.array(X_train).shape[1:])]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[35,70, 35], n_classes=4)

  # Train.
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_train}, y=y_train, num_epochs=100, shuffle=False)
classifier.train(input_fn=train_input_fn, steps=1000)

  # Predict.
test_input_fn = tf.estimator.inputs.numpy_input_fn(x={X_FEATURE: X_test}, y=y_test, num_epochs=1, shuffle=False)
predictions = classifier.predict(input_fn=test_input_fn)
y_predicted = np.array(list(p['class_ids'] for p in predictions))
y_predicted = y_predicted.reshape(np.array(y_test).shape)

  # Score with sklearn.
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
scores = classifier.evaluate(input_fn=test_input_fn)
print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))

classifier.predict()
#if __name__ == '__main__':
   # tf.app.run()