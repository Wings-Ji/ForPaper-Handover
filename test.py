import createData_for_knn as cd
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import DecisionTreeClassifier as dclf

data = cd.createData(5000)
x,y = data[:,:-1],data[:,-1]
# dataSet, labels,testVec,testVec_label =tree.createDataSet(20000)

MyDeciTree = DecisionTreeClassifier()
print('DecisionTree:',end='')
print(cross_val_score(MyDeciTree,x,y,cv=3))

bestTree = dclf.fit_model(x,y)
print('DecisionTree(best para):',end='')
print(cross_val_score(bestTree,x,y,cv=3))


k_Neighbors_clf =KNeighborsClassifier(5)
k_Neighbors_clf.fit(x,y)
print('k-Neighbors:',end='')
print(cross_val_score(k_Neighbors_clf,x,y,cv=3))

k_best = dclf.k_neighbors_best_estimator(x,y)
print('k-Neighbors(best para):',end='')
print(cross_val_score(k_best,x,y,cv=3))



