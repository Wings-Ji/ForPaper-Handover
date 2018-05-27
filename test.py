import tree
import createData_for_knn as cd
from sklearn.cross_validation import train_test_split

dataSet, labels,testVec,testVec_label =tree.createDataSet(2000)

featLabels = labels[:]
mytree = tree.createTree(dataSet,labels)
positive_result_count = 0.0
for i in range(len(testVec)):
    predicted = tree.classify(mytree,featLabels,testVec[i])
    print('predict result :',end='')
    print(predicted)
    print('True result is :',end='')
    print(testVec_label[i])
    if predicted == testVec_label[i]:
        positive_result_count += 1
print('acc:'+str(positive_result_count/len(testVec)))

