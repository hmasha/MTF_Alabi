#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 02:48:37 2019
https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
@author: hillary.masha
"""
#for vertical images cap the number of pixels read
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.image as mpimg

data = pd.read_csv("/Users/hillary.masha/output_classifier.csv")
test = pd.read_csv("/Users/hillary.masha/mtf_test_data.csv")
dataset = test.values
Z_test = test.drop('Class', axis = 1)
h_test = test["Class"]
#print("data_info\n")
#print(data.info())
print("\nData Shape(Number of rows, Number of columns)")
print(data.shape)
print("\nSnippet of data being processed to train decision tree")
print(data.head() )

X = data.drop('Class',axis = 1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#print tree
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))



#test on new images
y_pred = classifier.predict(Z_test)
print("Test data")
print(Z_test)
print("Test correct result(Expected result)")
print(h_test)
i = 0
for y in h_test:
    if h_test[i] == 1:
        print("Pass")
    else:
        print("Fail")
    i = i+1

print("\nPrediction Matrix")
print(y_pred)
print("\nConfusion Matrix")
print(confusion_matrix(h_test, y_pred))
print("\nClassification Report")
print(classification_report(h_test, y_pred))

print("\nDecision tree algorithm prediction")
i = 0
for y in y_pred:
    if y_pred[i] == 1:
        print("Pass")
    else:
        print("Fail")
    i = i+1


