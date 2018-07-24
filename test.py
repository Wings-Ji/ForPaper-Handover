from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
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
    svm_clf = svm.SVC(C=110,gamma='auto')
    c_range = np.logspace(-2,10,13)
    gamma_range = np.logspace(-9,3,13)

    params = dict(gamma=gamma_range)
    grid = GridSearchCV(svm_clf,params)
    grid.fit(x,y)
    print('svm start ...')
    print(grid.best_params_)
    return grid.best_estimator_

def MLPClassifier_estimator(x,y):
    clf = MLPClassifier(hidden_layer_sizes=(128, 7),random_state=1)
    # clf.fit(x, y)
    params = {'hidden_layer_sizes': [(500,),(128, 7),(50,50)],
              'alpha':[0.0001,0.003],'solver':['adam','lbfgs']}
    grid = GridSearchCV(clf, param_grid=params)
    grid.fit(x, y)
    print('MLPClassifier ...')
    print(grid.best_params_)
    return clf  # grid.best_estimator_

def GaussianNB_estimator(x,y):
    gnb = GaussianNB()
    # params = {}
    # grid = GridSearchCV(gnb,param_grid=params)
    # grid.fit(x,y)
    # print('GaussianNB ...')
    # print(grid.best_params_)
    gnb.fit(x,y)
    return gnb

def RandomForest_best_estimator(X,y):
    rf = RandomForestClassifier(n_estimators=42, criterion='gini', max_depth=44)
    params = dict(max_depth=[i for i in range(10,50,8)],n_estimators = [i for i in range(10,30,4)])  # 'max_depth':range(40,60,2),'n_estimators':range(26,50,2)
    print('RandomForest start search...')
    grid = GridSearchCV(rf, params)
    grid.fit(X, y)
    print(grid.best_params_)
    # print(grid.best_score_)
    rf_best = grid.best_estimator_
    print(cross_val_score(rf_best, X, y, cv=3))
    return rf_best

if __name__ == '__main__':
    # data = data_normalized.createData(8000)
    data = createData.createData(600)
    x,y = data[:, :-1], data[:, -1]

    print(cross_val_score(MLPClassifier_estimator(x,y),x,y,cv=2))