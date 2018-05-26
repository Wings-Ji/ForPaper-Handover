from numpy import *
import operator
import createData_for_knn as cd
from sklearn.cross_validation import train_test_split
from sklearn import metrics

def classify(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = argsort(distances)
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

data = cd.createData(10000)              #[00000100010000010010]
features = data[:,0:-1]
labels = data[:,-1]
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.33, random_state=23323)
predict_label = []
for i in range(len(test_features)):
    result = classify(test_features[i],train_features,train_labels,5)
    predict_label.append(result)
    if i%100 == 0:
        print(str(i)+' test label result is :',end='')
        print(test_labels[i],end='')
        print(', predict result is :',end='')
        print(result)
predict_label = array(predict_label)

print('confusion_matrix:')
print(metrics.confusion_matrix(test_labels,predict_label))

print('precision: ',end='')
print(metrics.precision_score(test_labels,predict_label,average=None))

print('recall:',end='')
print(metrics.recall_score(test_labels,predict_label,average=None))

print('accu:',end='')
print(metrics.accuracy_score(test_labels,predict_label))

print('classify_report:')
print(metrics.classification_report(test_labels,predict_label))