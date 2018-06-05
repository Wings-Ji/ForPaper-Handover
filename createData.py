# -*- coding: utf-8 -*-
# @Time    : 18/6/3 下午 9:29
# @Author  : Ji
# @File    : createData.py
# @Software: PyCharm
# 采集的数据首先进行离散化；
# 数据特征可以包含：星期(2) 天气(2) 时间(6) 行进方向(4) 离基站方向(6) 前一个BS(6)| (信号强度，性别，年龄等)
import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

random.seed(2)

def createData(DATANUM, IsNumpy = True, CREATE_CSV = False):
    SourceData = []
    for _ in range(DATANUM):
        SourceData_single = list(np.zeros(26))
        week = random.randint(0, 1)
        weather = random.randint(0, 1)
        time = random.randint(0, 5)
        move_direction = random.randint(0,3)
        location_of_BS = random.randint(0,5)
        last_BS = random.randint(0,5)

        SourceData_single[week] = 1
        SourceData_single[weather + 2] = 1
        SourceData_single[time + 4] = 1
        SourceData_single[move_direction + 10] = 1
        SourceData_single[location_of_BS + 14] = 1
        SourceData_single[last_BS + 20] = 1

        SourceData.append(SourceData_single)

    #根据实际行为特征分配label:
    allocat_sd(SourceData)

    #转为numpy矩阵
    SourceData_array = np.array(SourceData)
    # #特征标准化之后和标签拼接
    # feature_array = SourceData_array[:,:-1]
    # label_array = SourceData_array[:,-1]
    # feature_scaled = preprocessing.scale(feature_array)#标准化特征
    # SourceData_array = np.column_stack((feature_scaled,label_array))
    # 写入
    if CREATE_CSV == True:
        # featrueset = ['week','weather','time','directions']
        with open('dataSet.csv', 'w', newline='') as f:
            Iter_writer = csv.writer(f)
            # Iter_writer.writerow(featrueset)
            for sd in list(SourceData_array):
                Iter_writer.writerow(sd)
    if IsNumpy :
        return SourceData_array
    else:
        return SourceData        #==list(SourceData_array)

def allocat_sd(SourceData):
    for sd in SourceData:
        week_index, weather_index, time_index, move_direction_index, \
        location_of_BS_index, last_BS_index = find_index(sd)
        if week_index == 0:  # 周末
            if weather_index == 0:  #good weather
                if move_direction_index == 1 or move_direction_index == 0:
                    sd.append(find_direction(1,time_index,location_of_BS_index, last_BS_index))
                else:
                    sd.append(find_direction(2,time_index, location_of_BS_index, last_BS_index))
            elif weather_index == 1:  #bad weather
                if move_direction_index == 1 or move_direction_index == 0:
                    sd.append(find_direction(4,time_index,location_of_BS_index, last_BS_index))
                else:
                    sd.append(find_direction(5,time_index, location_of_BS_index, last_BS_index))
        else:               #周内
            if weather_index == 0:  #good weather
                if move_direction_index == 1 or move_direction_index == 0:
                    sd.append(find_direction(2,time_index,location_of_BS_index, last_BS_index))
                else:
                    sd.append(find_direction(3,time_index, location_of_BS_index, last_BS_index))
            elif weather_index == 1:  #bad weather
                if move_direction_index == 1 or move_direction_index == 0:
                    sd.append(find_direction(5,time_index,location_of_BS_index, last_BS_index))
                else:
                    sd.append(find_direction(6,time_index, location_of_BS_index, last_BS_index))

def find_direction(func_index,time_index,location_of_BS_index, last_BS_index):
    if func_index == 1 or func_index == 4:
        if location_of_BS_index == 1 and last_BS_index == 4:
            return 1
        elif location_of_BS_index == 2 and last_BS_index == 5:
            return 2
        elif location_of_BS_index == 3 and last_BS_index == 6:
            return 3
        else:
            return time_allocate(1,time_index)
    elif func_index == 2 :
        if location_of_BS_index == 4 and last_BS_index == 1:
            return 4
        elif location_of_BS_index == 5 and last_BS_index == 2:
            return 5
        elif location_of_BS_index == 6 and last_BS_index == 3:
            return 6
        else:
            return time_allocate(2,time_index)
    elif func_index == 3:
        return time_allocate(3, time_index)
    elif func_index == 5:
        return time_allocate(4,time_index)
    else:
        return time_allocate(5, time_index)

def time_allocate(func_index, time):
    if func_index == 1:
        if time == 0:
            return 2
        elif time == 1:
            return 1
        elif time == 2:
            return 3
        elif time == 3:
            return 4
        elif time == 4:
            return 5
        elif time == 5:
            return random.randint(1,6)
    if func_index == 2:
        if time == 0:
            return 3
        elif time == 1:
            return 2
        elif time == 2:
            return 2
        elif time == 3:
            return 2
        elif time == 4:
            return 5
        elif time == 5:
            return random.randint(4, 6)
    if func_index == 3:
        if time == 0:
            return 5
        elif time == 1:
            return 1
        elif time == 2:
            return 2
        elif time == 3:
            return 3
        elif time == 4:
            return 4
        elif time == 5:
            return random.randint(3, 5)
    if func_index == 4:
        if time == 0:
            return 4
        elif time == 1:
            return 1
        elif time == 2:
            return 3
        elif time == 3:
            return 5
        elif time == 4:
            return 6
        elif time == 5:
            return random.randint(2, 4)
    if func_index == 5:
        if time == 0:
            return 6
        elif time == 1:
            return 2
        elif time == 2:
            return 1
        elif time == 3:
            return 6
        elif time == 4:
            return 4
        elif time == 5:
            return random.randint(2, 4)

def find_index(list):
    for i in range(0,2):
        if list[i] > 0:
            week = i
    for i in range(2,4):
        if list[i] > 0:
            weather = i- 2
    for i in range(4,10):
        if list[i] > 0:
            time = i-4
    for i in range(10, 14):
        if list[i] > 0:
            move_direction = i - 10
    for i in range(14, 20):
        if list[i] > 0:
            location_of_BS = i - 14
    for i in range(20, 26):
        if list[i] > 0:
            last_BS = i - 20

    return week, weather, time, move_direction, location_of_BS, last_BS

def createdata_split_it(dataVolume):
    data = createData(dataVolume)
    features = data[:, 0:-1]
    data_labels = data[:, -1]
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, data_labels, test_size=0.33, random_state=23323)
    return train_features, test_features, train_labels, test_labels

if __name__ == '__main__':
    data = createData(200,CREATE_CSV=True)