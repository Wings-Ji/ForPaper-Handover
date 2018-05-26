import random
import numpy as np
import csv
#
def createData(DATANUM):
    SourceData = []
    for _ in range(DATANUM):
        SourceData_week = random.randint(0, 6)
        SourceData_weather = random.randint(1, 4)
        SourceData_time = random.uniform(0,23)
        # SourceData[SourceData_week] = 1
        # SourceData[SourceData_weather - 1 + 7] = 1
        # SourceData[SourceData_time + 11] = 1
        SourceData.append([SourceData_week,SourceData_weather,SourceData_time])
        # SourceData.append(SourceData_single)

    for sd in SourceData:
        if sd[0]==0 or sd[0]==6:   #周末
            if sd[1]==1 or sd[1]==4:   #1或4天气，即晴或凤
                sd.append(direction_4(sd[2]))
            elif sd[1]==2 or sd[1]==3:
                sd.append(direction_5(sd[2]))
        else:   #工作日
            if sd[1]==1:   #天气1，即晴天
                sd.append(direction_1(sd[2]))
            elif sd[1]==2:
                sd.append(direction_2(sd[2]))
            elif sd[1]==3 or sd[1]==4:
                sd.append(direction_3(sd[2]))
    # writefile(SourceData)
    SourceData = np.array(SourceData)

    return SourceData

def writefile(sData):

    with open('createdData.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(sData)


def direction_1(time):
    if time < 9.0 and time > 5.0:
        return 1
    elif time >11.0 and time < 12.0:
        return 3
    elif time > 12.0 and time < 14.0:
        return 6
    elif time > 16.0 and time < 22.0:
        return 4
    else:
        return 2# return random.randint(1,6)

def direction_2(time):
    if time < 9.0 and time > 5.0:
        return 1
    elif time >11.0 and time < 12.0:
        return 6
    elif time > 12.0 and time < 14.0:
        return 3
    elif time > 16.0 and time < 22.0:
        return 4
    else:
        return 3

def direction_3(time):
    if time < 9.0 and time > 5.0:
        return 2
    elif time >11.0 and time < 12.0:
        return 6
    elif time > 12.0 and time < 14.0:
        return 5
    elif time > 16.0 and time < 22.0:
        return 4
    else:
        return 4

def direction_4(time):
    if time < 9.0 and time > 5.0:
        return 1
    elif time >11.0 and time < 12.0:
        return 6
    elif time > 12.0 and time < 14.0:
        return 5
    elif time > 16.0 and time < 22.0:
        return 4
    else:
        return 5

def direction_5(time):
    if time < 9.0 and time > 5.0:
        return 1
    elif time >11.0 and time < 12.0:
        return 2
    elif time > 12.0 and time < 14.0:
        return 3
    elif time > 16.0 and time < 22.0:
        return 4
    else:
        return 6

# print(createData(100))