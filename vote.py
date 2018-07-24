# -*- coding: utf-8 -*-
# @Time    : 18/6/7 下午 3:52
# @Author  : Ji
# @File    : vote.py
# @Software: PyCharm

import Classifiers
import createData
import time
from sklearn.model_selection import train_test_split
import random

def clf_vote(dataVolume,no_rand_rate=0):
    data = createData.createData(dataVolume)  # 星期(2) 天气(2) 时间(6) 行进方向(4) 离基站方向(6) 前一个BS(6)
    x, y = data[:, :-1], data[:, -1]
    if no_rand_rate != 0 :
        for i in range(len(y)):
            if y[i] % no_rand_rate == 0:
                y[i] = random.randint(1, 6)
    X_train, X_test, y_train, y_test = train_test_split(x , y, test_size = 0.5, random_state = 42)
    svm_clf = Classifiers.svm_estimator(X_train,y_train)
    RF_clf = Classifiers.RandomForest_best_estimator(X_train,y_train)
    DT_clf = Classifiers.DecisionTree_best_estimator(X_train,y_train)
    mlp_clf = Classifiers.MLPClassifier_estimator(X_train,y_train)
    count = 0.0
    time_start = time.time()
    pre_svm = list(svm_clf.predict(X_test))
    pre_RF = list(RF_clf.predict(X_test))
    pre_DT = list(DT_clf.predict(X_test))
    pre_MLP = list(mlp_clf.predict(X_test))
    time_end = time.time()
    time_pre = time_end - time_start
    for i in range(len(pre_svm)):
        vote_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        vote_dict[pre_svm[i]] += 1
        vote_dict[pre_RF[i]] += 1
        vote_dict[pre_DT[i]] += 1
        vote_dict[pre_MLP[i]] += 1
        vote_result = max(vote_dict, key=vote_dict.get)
        if vote_result == y_test[i]:
            count +=1
    acc = count / len(y_test)
    return time_pre,acc


if __name__ == '__main__':
    timepre, acc = clf_vote(1000)
    print('result :')
    print(timepre,acc)
