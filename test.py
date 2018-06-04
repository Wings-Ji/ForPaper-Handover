from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import numpy as np
import createData
from sklearn.neural_network import MLPClassifier
import data_normalized
import csv
import createData_for_knn as cd
import random
from sklearn import svm

def DecisionTree_best_estimator(X,y):
    # k_fold = 1
    MyDecisionTree = DecisionTreeClassifier(random_state=0)
    # params ={'max_depth':range(10,40),'criterion':np.array(['entropy','gini'])}
    # grid = GridSearchCV(MyDecisionTree,params)
    # print('DecisionTree start ...')
    # grid.fit(X,y)
    # return grid.best_estimator_
    MyDecisionTree.fit(x,y)
    return MyDecisionTree

def svm_estimator(x,y):
    svm_clf = svm.SVC(C=0.4,gamma='auto')
    c_range = np.logspace(-2,10,13)
    gamma_range = np.logspace(-9,3,13)

    params = dict(gamma = gamma_range,C = c_range)
    grid = GridSearchCV(svm_clf,params)
    grid.fit(x,y)
    print(grid.best_params_)
    return grid.best_estimator_

def MLPClassifier_estimator(x,y):
    clf = MLPClassifier()
    params = {'hidden_layer_sizes': [(7, 7), (128,), (128, 7)],
              'solver': ['lbfgs', 'sgd', 'adam'],'alpha':[0.0001,0.0005]}
    grid = GridSearchCV(clf, param_grid=params)
    grid.fit(x, y)
    print('MLPClassifier ...')
    print(grid.best_params_)
    return grid.best_estimator_

if __name__ == '__main__':
    # data = data_normalized.createData(8000)
    data = createData.createData(5000)
    x,y = data[:, :-1], data[:, -1]

    print(cross_val_score(MLPClassifier_estimator(x,y),x,y))