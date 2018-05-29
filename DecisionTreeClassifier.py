import createData_for_knn as cd
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from IPython.display import Image
import pydotplus

data = cd.createData(1000)

def fit_model(X,y):
    k_fold = 3
    MyDecisionTree = DecisionTreeClassifier(random_state=0)
    params ={'max_depth':range(10,40),'criterion':np.array(['entropy','gini'])}
    grid = GridSearchCV(MyDecisionTree,params,cv=k_fold)
    grid.fit(X,y)
    return grid.best_estimator_

def plotTree(clf):
    dot_data = tree.export_graphviz(clf,out_file=None)#out_file=None,filled=True, rounded=True,special_characters=True
    graph = pydotplus.graph_from_dot_data(dot_data)
    # graph.write_png('tree.png')
    Image(graph.create_png())

def k_neighbors_best_estimator(x,y):
    k_Neighbors_clf = KNeighborsClassifier()
    # k_Neighbors_clf.score()
    params = {'n_neighbors':range(1,8),'leaf_size':range(20,50)}
    grid = GridSearchCV(k_Neighbors_clf,params)
    grid.fit(x,y)
    return grid.best_estimator_


if __name__ == '__main__':
    bestTree = fit_model(data[:,:-1],data[:,-1])
    print(cross_val_score(bestTree,data[:,:-1],data[:,-1],cv=3))

    k_best = k_neighbors_best_estimator(data[:,:-1],data[:,-1])
    print(cross_val_score(k_best, data[:, :-1], data[:, -1], cv=3))
    # plotTree(bestTree)