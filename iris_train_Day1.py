# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:42:52 2018

@author: mai.mm.2
"""
#https://www.youtube.com/watch?v=N9fDIAflCMY

from sklearn import tree
import numpy as np
from sklearn.datasets import load_iris

features = [[140,1],[130,1],[150,0],[170,0]]# weight,type 
labels = [0,0,1,1]#0=apple,1=orange

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)
print clf.predict([[150,0]])

#input iris as a example

iris = load_iris()
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


