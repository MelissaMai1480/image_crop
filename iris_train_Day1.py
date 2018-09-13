# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:42:52 2018

@author: mai.mm.2
"""
#https://www.youtube.com/watch?v=N9fDIAflCMY

from sklearn import tree
import numpy as np
from sklearn.datasets import load_iris
from sklearn import datasets

features = [[140,1],[130,1],[150,0],[170,0]]# weight,type 
labels = [0,0,1,1]#0=apple,1=orange

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print clf.predict([[150,0]])

#input iris as a example

iris = datasets.load_iris()
print iris.feature_names
print iris.target_names
print iris.data[0]
print iris.target[0]
test_idx = [0,50,100]
#traing data
train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis=0)


#test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

#预测测试数据
print test_target
print clf.predict(test_data)

#viz code
from sklearn.externals.six import StringIO
import pydot
dot_data = StringIO()
tree.export_graphviz(clf,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True,rounded=True,
                         impurity=False)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('iris.pdf')

print test_data[0],test_target[0]
print iris.feature_name,iris.target_names

#greyhounds dogs

import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

plt.hist([grey_height,lab_height],stacked=True,color=['r','b'])
plt.show()


#import datasets
from sklearn import datasets



iris = datasets.load_iris()
X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5)

#分类器1
#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()

#分类器2
#from sklearn.neighbors import  KNeighborsClassifier
#my_classifier = KNeighborsClassifier()

from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)
#defind clf
class ScrappyKNN():
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self,X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self,row):
        best_dist = euc(row,self.X_train[0])
        best_index = 0
        for i in range(1,len(self.X_train)):
            dist = euc(row,self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]
    
    
my_classifier = ScrappyKNN()


my_classifier.fit(X_train,y_train)

predictions = my_classifier.predict(X_test)
print predictions

from sklearn.metrics import accuracy_score
print accuracy_score(y_test,predictions)


#p6
from sklearn import metrics,cross_validation
import tensorflow as tf
from tensorflow.contrib import learn
 
def main(unsure_argv):
    #load datasets
    iris = learn.datasets.load_datasets('iris')
    x_train,x_test,y_train,y_test = cross_validation.train_test_split(
            iris.data,iris.target,test_size=0.2,random_state=42)
    
    #built 3 layer DNN with 10,20,10 units respectively
    classifier = learn.DNNClassifier(hiddden_units=[10,20,10],n_classes=3)
    
    #fit and predict
    classifier.fit(x_train,y_train,steps=200)
    score = metrics.accuracy_score(y_test,classifier.predict(x_test))
    print('Accuracy:{0:1}'.format(score))

