import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = svm.SVC()
# .fit can be thought of as .train 
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

# example_measures = np.array([[8,2,1,4,4,2,3,2,1],[8,2,1,2,8,2,3,2,6]])
# print(example_measures)
# example_measures = example_measures.reshape(len(example_measures), -1)
# print(example_measures)
# prediction = clf.predict(example_measures)
# print(prediction)
