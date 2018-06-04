#本数据包含星期、天气和时间 数据内容和createData_for_knn中的数据相同
#只是格式上的不同，生成的数据最后进行了标准化处理
import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

random.seed(2)

def createData(DATANUM, IsNumpy = True, CREATE_CSV = False):
    SourceData = []
    for _ in range(DATANUM):
        SourceData_week = random.randint(0, 6)
        SourceData_weather = random.randint(0, 3)
        SourceData_time = random.randint(0, 23)
        # SourceData_others = random.randint(0,10)
        SourceData.append([SourceData_week,SourceData_weather,SourceData_time])#,SourceData_others
    #for each label :
    for sd in SourceData:
        week, weather, time = sd[0],sd[1],sd[2]
        if week == 0 or week == 6:  # 周末
            if weather == 0:
                sd.append(direction_1(time))
            elif weather == 1:
                sd.append(direction_2(time))
            elif weather == 2 or weather ==3:
                sd.append(direction_3(time))
        else:
            if weather == 0:
                sd.append(direction_4(time))
            elif weather == 1:
                sd.append(direction_5(time))
            elif weather == 2 or weather == 3:
                sd.append(direction_6(time))

    #转为numpy矩阵
    SourceData_array = np.array(SourceData)
    #特征标准化之后和标签拼接
    feature_array = SourceData_array[:,:-1]
    label_array = SourceData_array[:,-1]
    feature_scaled = preprocessing.scale(feature_array)#标准化特征
    data_scale = np.column_stack((feature_scaled,label_array))
    # 写入
    if CREATE_CSV == True:
        featrueset = ['week','weather','time','directions']
        with open('dataSet.csv', 'w', newline='') as f:
            Iter_writer = csv.writer(f)
            Iter_writer.writerow(featrueset)
            for sd in list(data_scale):
                Iter_writer.writerow(sd)
    if IsNumpy :
        return data_scale
    else:
        return list(data_scale)

def find_index(list):
    for i in range(0,7):
        if list[i] > 0:
            week = i
    for i in range(7,11):
        if list[i] > 0:
            weather = i-7
    for i in range(11,35):
        if list[i] > 0:
            time = i-11
    return week,weather,time

def direction_1(time):
    if time > 0.0 and time <5.0:
        return 2
    elif time > 5.0 and time < 10.0:
        return 1
    elif time > 11.0 and time < 14.0:
        return 3
    elif time > 16.0 and time < 21.0:
        return 4
    elif time > 21.0 and time < 23.0:
        return 6
    else:
        return random.randint(4,6)

def direction_2(time):
    if time > 0.0 and time <5.0:
        return 1
    elif time > 5.0 and time < 10.0:
        return 1
    elif time > 11.0 and time < 14.0:
        return 2
    elif time > 16.0 and time < 21.0:
        return 5
    elif time > 21.0 and time < 23.0:
        return 6
    else:
        return random.randint(4,6)

def direction_3(time):
    if time > 0.0 and time <5.0:
        return 2
    elif time > 5.0 and time < 10.0:
        return 2
    elif time > 11.0 and time < 14.0:
        return 3
    elif time > 16.0 and time < 21.0:
        return 5
    elif time > 21.0 and time < 23.0:
        return 6
    else:
        return random.randint(1,3)

def direction_4(time):
    if time > 0.0 and time <5.0:
        return 3
    elif time > 5.0 and time < 10.0:
        return 1
    elif time > 11.0 and time < 14.0:
        return 4
    elif time > 16.0 and time < 21.0:
        return 4
    elif time > 21.0 and time < 23.0:
        return 5
    else:
        return random.randint(1,3)

def direction_5(time):
    if time > 0.0 and time <5.0:
        return 3
    elif time > 5.0 and time < 10.0:
        return 6
    elif time > 11.0 and time < 14.0:
        return 3
    elif time > 16.0 and time < 21.0:
        return 4
    elif time > 21.0 and time < 23.0:
        return 6
    else:
        return random.randint(1,3)

def direction_6(time):
    if time > 0.0 and time <5.0:
        return 2
    elif time > 5.0 and time < 10.0:
        return 5
    elif time > 11.0 and time < 14.0:
        return 3
    elif time > 16.0 and time < 21.0:
        return 5
    elif time > 21.0 and time < 23.0:
        return 6
    else:
        return random.randint(1,3)

def createdata_split_it(dataVolume):
    data = createData(dataVolume)
    features = data[:, 0:-1]
    data_labels = data[:, -1]
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, data_labels, test_size=0.33, random_state=23323)
    return train_features, test_features, train_labels, test_labels

if __name__ == '__main__':
    data = createData(200,CREATE_CSV=True)