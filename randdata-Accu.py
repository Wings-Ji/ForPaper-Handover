# -*- coding: utf-8 -*-
# @Time    : 18/6/25 下午 4:02
# @Author  : Ji
# @File    : randdata-Accu.py
# @Software: PyCharm
import Classifiers as clf
import time
import createData
from sklearn.model_selection import cross_val_score
import vote
import random
import matplotlib.pyplot as plt
import numpy as np

def dataVolume_runtime(dataVolume, no_rand_rate = 2 ):
    # data = cd.createData(dataVolume)
    data = createData.createData(dataVolume) #星期(2) 天气(2) 时间(6) 行进方向(4) 离基站方向(6) 前一个BS(6)
    # data = data_normalized.createData(dataVolume)
    x, y = data[:, :-1], data[:, -1]
    for i in range(len(y)):
        if y[i]%no_rand_rate == 0:
            y[i] = random.randint(1,6)
    print('data set done!')


    #DecisionTree:
    bestTree = clf.DecisionTree_best_estimator(x, y)
    DTScore = cross_val_score(bestTree, x, y)

    #k-Neighbors:
    k_best = clf.k_neighbors_best_estimator(x, y)
    k_Neighbors_score = cross_val_score(k_best, x, y)

    #randomforest:
    rf_best= clf.RandomForest_best_estimator(x,y)
    rf_score = cross_val_score(rf_best, x, y)

    #svm:
    svm_model = clf.svm_estimator(x,y)
    svm_score = cross_val_score(svm_model,x,y)

    #MLPClassifier
    nn_best = clf.MLPClassifier_estimator(x,y)
    nn_score = cross_val_score(nn_best,x,y,cv=6)

    #vote
    vote_time,vote_acc = vote.clf_vote(dataVolume*2,no_rand_rate = no_rand_rate)

    # return DTScore,k_Neighbors_score,rf_score,svm_score,dttime,knntime,rftime,svmtime
    return float(sum(DTScore)) / len(DTScore),float(sum(k_Neighbors_score)) / len(k_Neighbors_score),\
           float(sum(rf_score)) / len(rf_score),float(sum(svm_score)) / len(svm_score), \
           float(sum(nn_score)) / len(nn_score),vote_acc,vote_time

def plot_randdata_accu(volume):
    no_rand_rate_range = [50,54,59,63,68,72,77,81,85,90]#range(0.5,0.9)
    dtset,knnset,rfset,svmset,nnset,voteset = [],[],[],[],[],[]
    for i in range(2,12):
        DTScore, k_Neighbors_score, rf_score, svm_score, nn_score, vote_score, \
        votetime = dataVolume_runtime(volume,no_rand_rate=i)
        dtset.append(DTScore)
        knnset.append(k_Neighbors_score)
        rfset.append(rf_score)
        svmset.append(svm_score)
        nnset.append(nn_score)
        voteset.append(vote_score)
        print(DTScore, k_Neighbors_score, rf_score, svm_score, nn_score,vote_score)
    plt.plot(no_rand_rate_range, dtset, 'r^-', label='DecisionTree')
    # plt.plot(no_rand_rate_range, knnset, 'yo-', label='k-Neighbors')
    plt.plot(no_rand_rate_range, rfset, 'gs-', label='RandomForest')
    plt.plot(no_rand_rate_range, svmset, 'bp-', label='SVM')
    plt.plot(no_rand_rate_range, nnset, 'mv-', label='neural_network')
    plt.plot(no_rand_rate_range, voteset, 'kv-.', label='Proposed')  # vote
    plt.xlabel('NoRand_rate/%')
    plt.ylabel('Accuracy')
    plt.xticks()
    plt.legend(loc=2, fontsize=8)
    plt.show()

if __name__ == '__main__':
    plot_randdata_accu(1300)