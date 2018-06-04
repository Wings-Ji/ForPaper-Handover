from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
import createData_for_knn as cd

def runthis(datavolume):
    data = cd.createData(datavolume)
    x, y = data[:, :-1], data[:, -1]
    rf = RandomForestClassifier(n_estimators=42,criterion='gini',max_depth=44)
    params = {'criterion':['gini','entropy']}#'max_depth':range(40,60,2),'n_estimators':range(26,50,2)
    print('start search')
    grid = GridSearchCV(rf,params,cv=3)
    grid.fit(x,y)
    print(grid.best_params_)
    print(grid.best_score_)
    rf_best = grid.best_estimator_

    print(cross_val_score(rf_best,x,y,cv=3))


if __name__ == '__main__':
    runthis(1000)