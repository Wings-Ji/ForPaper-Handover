import time
import data_normalized
import createData
import matplotlib.pyplot as plt
import createData_for_knn as cd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree,svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from IPython.display import Image
import pydotplus

def DecisionTree_best_estimator(X,y):
    # k_fold = 1
    MyDecisionTree = DecisionTreeClassifier(random_state=0)
    params ={'max_depth':range(10,40),'criterion':np.array(['entropy','gini'])}
    grid = GridSearchCV(MyDecisionTree,params)
    print('DecisionTree start ...')
    grid.fit(X,y)
    return grid.best_estimator_

def RandomForest_best_estimator(X,y):
    rf = RandomForestClassifier(n_estimators=26, criterion='gini', max_depth=18)
    params =dict(n_estimators = [24,26,28],max_depth = [14,18,22])  # 'max_depth':range(40,60,2),'n_estimators':range(26,50,2)
    print('RandomForest start search...')
    grid = GridSearchCV(rf, params)
    grid.fit(X, y)
    # print(grid.best_params_)
    # print(grid.best_score_)
    rf_best = grid.best_estimator_
    print(cross_val_score(rf_best, X, y, cv=3))
    return rf_best

def k_neighbors_best_estimator(x,y):
    k_Neighbors_clf = KNeighborsClassifier()
    # k_Neighbors_clf.score()
    params = {'n_neighbors':range(1,8),'leaf_size':range(20,50)}
    grid = GridSearchCV(k_Neighbors_clf,params)
    grid.fit(x,y)
    return grid.best_estimator_

def svm_estimator(x,y):
    svm_clf = svm.SVC(C=110, kernel='rbf',gamma='auto')
    svm_clf.fit(x, y)
    print('svm start...')
    print(cross_val_score(svm_clf, x, y, cv=3))
    return svm_clf

def MLPClassifier_estimator(x,y):
    clf = MLPClassifier(solver = 'adam')
    params = {'hidden_layer_sizes': [(130,50,7),(128, 7)],
              'alpha':[0.0025,0.003]}
    grid = GridSearchCV(clf, param_grid=params)
    grid.fit(x, y)
    print('MLPClassifier ...')
    print(grid.best_params_)
    return grid.best_estimator_

def dataVolume_runtime(dataVolume):
    # data = cd.createData(dataVolume)
    data = createData.createData(dataVolume) #星期(2) 天气(2) 时间(6) 行进方向(4) 离基站方向(6) 前一个BS(6)
    # data = data_normalized.createData(dataVolume)
    x, y = data[:, :-1], data[:, -1]

    #DecisionTree:
    bestTree = DecisionTree_best_estimator(x, y)
    time1Start = time.time()
    DTScore = cross_val_score(bestTree, x, y)
    time1End = time.time()
    dttime = time1End - time1Start

    #k-Neighbors:
    k_best = k_neighbors_best_estimator(x, y)
    time2Start = time.time()
    k_Neighbors_score = cross_val_score(k_best, x, y)
    time2End = time.time()
    knntime = time2End - time2Start

    #randomforest:
    rf_best= RandomForest_best_estimator(x,y)
    time_rf_Start = time.time()
    rf_score = cross_val_score(rf_best, x, y)
    time_rf_End = time.time()
    rftime = time_rf_End - time_rf_Start

    #svm:
    svm_model = svm_estimator(x,y)
    time_svm_start = time.time()
    svm_score = cross_val_score(svm_model,x,y)
    time_svm_end = time.time()
    svmtime = time_svm_end-time_svm_start

    #MLPClassifier
    nn_best = MLPClassifier_estimator(x,y)
    time_nn_start = time.time()
    nn_score = cross_val_score(nn_best,x,y)
    time_nn_end = time.time()
    nntime = time_nn_end - time_nn_start


    # return DTScore,k_Neighbors_score,rf_score,svm_score,dttime,knntime,rftime,svmtime
    return float(sum(DTScore)) / len(DTScore),float(sum(k_Neighbors_score)) / len(k_Neighbors_score),\
           float(sum(rf_score)) / len(rf_score),float(sum(svm_score)) / len(svm_score), \
           float(sum(nn_score)) / len(nn_score),dttime,knntime,rftime,svmtime,nntime

def polt_estimators(startVolume,endVolume,span):#决策树、kNN、随机森林、svm
    dtscoreset,knnscoreset, RFscoreset,SVMscoreset,NNscoreset= [], [], [], [], []
    dttimeset,knntimeset, RFtimeset, SVMtimeset, NNtimeset = [], [], [], [], []
    for dataVolume in range(startVolume,endVolume,span):
        print('start times:' + str(dataVolume))
        DTScore, k_Neighbors_score, rf_score, svm_score, nn_score, \
        dttime, knntime, rftime, svmtime, nntime = dataVolume_runtime(dataVolume)
        dtscoreset.append(DTScore)
        knnscoreset.append(k_Neighbors_score)
        RFscoreset.append(rf_score)
        SVMscoreset.append(svm_score)
        NNscoreset.append(nn_score)

        dttimeset.append(dttime)
        knntimeset.append(knntime)
        RFtimeset.append(rftime)
        SVMtimeset.append(svmtime)
        NNtimeset.append(nntime)
    VolumesRange = [i for i in range(startVolume,endVolume,span)]
    # axes1 = plt.subplot(2,2,1)
    plt.figure(1)
    plt.plot(VolumesRange, dttimeset, 'r^-',label= 'DecisionTree')
    plt.plot(VolumesRange, knntimeset, 'yo-',label= 'k-Neighbors')
    plt.plot(VolumesRange, RFtimeset, 'gs-',label= 'RandomForest')
    plt.plot(VolumesRange, SVMtimeset, 'bp-',label = 'SVM')
    plt.plot(VolumesRange, NNtimeset, 'mv-', label='neural_network')
    plt.xlabel('DataVolume')
    plt.ylabel('PredictTime')
    plt.legend(loc=2,fontsize = 9)
    plt.figure(2)
    plt.plot(VolumesRange, dtscoreset, 'r^-', label= 'DecisionTree')
    plt.plot(VolumesRange, knnscoreset, 'yo-',label= 'k-Neighbors')
    plt.plot(VolumesRange, RFscoreset, 'gs-',label= 'RandomForest')
    plt.plot(VolumesRange, SVMscoreset, 'bp-',label = 'SVM')
    plt.plot(VolumesRange, NNscoreset, 'mv-', label='neural_network')
    # plt.ylim(0,1.3)
    plt.xlabel('DataVolume')
    plt.ylabel('PredictScores')
    plt.legend(loc=4,fontsize = 9)
    plt.show()

def plotTree(clf):
    dot_data = tree.export_graphviz(clf,out_file=None)#out_file=None,filled=True, rounded=True,special_characters=True
    graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_png('tree.png')
    Image(graph.create_png())



if __name__ == '__main__':
    polt_estimators(100,2500,300)
