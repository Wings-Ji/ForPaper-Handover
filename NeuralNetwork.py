# coding:utf-8
'''''
Created on 2018

@author: J
'''
import numpy as np
import createData_for_knn as cd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt

np.random.seed(23)

# 定义双曲函数和他们的导数
def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))


# 定义NeuralNetwork 神经网络算法
class NeuralNetwork:
    # 初始化，layes表示的是一个list，eg[10,10,3]表示第一层10个神经元，第二层10个神经元，第三层3个神经元
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layer.
        Should be at least two values
        :param activation: The activation function to be used. Can be
        "logistic" or "tanh"
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        # 循环从1开始，相当于以第二层为基准，进行权重的初始化
        for i in range(1, len(layers) - 1):
            # 对当前神经节点的前驱赋值
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            # 对当前神经节点的后继赋值
            self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    # 训练函数   ，X矩阵，每行是一个实例 ，y是每个实例对应的结果，learning_rate 学习率，
    # epochs，表示抽样的方法对神经网络进行更新的最大次数
    def fit(self, X, y, learning_rate=0.2, epochs=10000):
        X = np.atleast_2d(X)  # 确定X至少是二维的数据
        temp = np.ones([X.shape[0], X.shape[1] + 1])  # 初始化矩阵
        temp[:, 0:-1] = X  # adding the bias unit to the input layer
        X = temp
        y = np.array(y)  # 把list转换成array的形式

        for k in range(epochs):
            # 随机选取一行，对神经网络进行更新
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            # 完成所有正向的更新
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
                #
            error = y[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]

            # 开始反向计算误差，更新权重
            for l in range(len(a) - 2, 0, -1):  # we need to begin at the second to last layer
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    # 预测函数

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

def print_evaluate(test_labels,predict_label):
    print('confusion_matrix:')
    print(metrics.confusion_matrix(test_labels, predict_label))

    print('precision: ', end='')
    print(metrics.precision_score(test_labels, predict_label, average=None))

    print('recall:', end='')
    print(metrics.recall_score(test_labels, predict_label, average=None))

    print('accu:', end='')
    print(metrics.accuracy_score(test_labels, predict_label))

    print('classify_report:')
    print(metrics.classification_report(test_labels, predict_label))

def optLayer(layer):
    data = cd.createData(10000)
    features = data[:,0:-1]
    labels = data[:,-1]
    min_max_scaler = preprocessing.MinMaxScaler()
    labels = min_max_scaler.fit_transform(labels) #归一化

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.33, random_state=23323)

    nn = NeuralNetwork(layers=[35,layer,1])
    nn.fit(train_features,train_labels,0.1)
    predice_result_set = []
    for x in test_features:
        y_pre = nn.predict(x)
        predice_result_set.append(y_pre)

    test_labels = min_max_scaler.inverse_transform(test_labels)
    predice_result_set = min_max_scaler.inverse_transform(predice_result_set)#去归一化

    positive = 0
    for i in range(len(test_features)):
        # if test_labels[i] == [round(int(pr)) for pr in predice_result_set][i]:
        if abs(test_labels[i] - predice_result_set[i]) < 0.5:
            positive = positive +1
        # print('x_true:',end='')
        # print(test_labels[i],end='')
        # print(', y_pre:',end='')
        # print(predice_result_set[i])
    return positive/len(test_labels)
    # print('acc:'+str(positive/len(test_labels)) )
    # plt.plot(test_labels[:50])
    # plt.plot(predice_result_set[:50],'r')
    # plt.show()

# acc_set = []
# for i in range(10,50):
#     acc_set.append(optLayer(i))
# plt.plot(acc_set)
# plt.show()
print('accu:'+str(optLayer(24)))
