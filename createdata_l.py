import random
import numpy as np
import csv


def createData(DATANUM):
    SourceData = []
    for _ in range(DATANUM):
        SourceData_single = list(np.zeros(35))
        SourceData_week = random.randint(0, 6)
        SourceData_weather = random.randint(0, 3)
        SourceData_time = random.randint(0, 23)
        SourceData_single[SourceData_week] = 1
        SourceData_single[SourceData_weather + 7] = 1
        SourceData_single[SourceData_time + 11] = 1
        # SourceData.append([SourceData_week,SourceData_weather,SourceData_time])
        SourceData.append(SourceData_single)

    for sd in SourceData:
        week, weather, time = find_index(sd)
        if week == 0 or week == 6:  # 周末
            if weather == 0:
                if time > 1 and time < 17:
                    sd.append(6)
                else:
                    sd.append(0)
            elif weather == 1:
                if time>10 and time < 17:
                    sd.append(3)
                else:
                    sd.append(0)
            elif weather == 2 or weather ==3:
                if time >0 and time<10:
                    sd.append(2)
                elif time > 17 and time < 24:
                    sd.append(5)
                else:
                    sd.append(0)
        else:
            if weather == 0:
                if time > 0 and time < 10:
                    sd.append(1)
                elif time > 17 and time < 24:
                    sd.append(4)
                else:
                    sd.append(0)
            elif weather == 1:
                if time > 10 and time < 17:
                    sd.append(3)
                else:
                    sd.append(0)
            elif weather == 2 or weather == 3:
                if time > 0 and time < 10:
                    sd.append(2)
                elif time > 17 and time < 24:
                    sd.append(5)
                else:
                    sd.append(0)

    SourceData = np.array(SourceData)
    # writefile(SourceData)
    return SourceData

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




def writefile(sData):
    with open('createdData.csv', 'w', newline='') as f:
        f.write(sData.tostring())



def direction_1(time):
    if time < 4.0 and time > 0.0:
        return 1
    elif time > 4.0 and time < 8.0:
        return 2
    elif time > 8.0 and time < 12.0:
        return 3
    elif time > 12.0 and time < 16.0:
        return 4
    elif time > 16.0 and time < 20.0:
        return 5
    elif time > 20.0 and time < 24.0:
        return 6
    else:
        return random.randint(1,6)

def direction_2(time):
    if time < 4.0 and time > 0.0:
        return 4
    elif time > 4.0 and time < 8.0:
        return 2
    elif time > 8.0 and time < 12.0:
        return 3
    elif time > 12.0 and time < 16.0:
        return 1
    elif time > 16.0 and time < 20.0:
        return 5
    elif time > 20.0 and time < 24.0:
        return 6
    else:
        return random.randint(1,6)

def direction_3(time):
    if time < 4.0 and time > 0.0:
        return 4
    elif time > 4.0 and time < 8.0:
        return 3
    elif time > 8.0 and time < 12.0:
        return 2
    elif time > 12.0 and time < 16.0:
        return 1
    elif time > 16.0 and time < 20.0:
        return 6
    elif time > 20.0 and time < 24.0:
        return 5
    else:
        return random.randint(1,6)

def direction_4(time):
    if time < 4.0 and time > 0.0:
        return 2
    elif time > 4.0 and time < 8.0:
        return 3
    elif time > 8.0 and time < 12.0:
        return 1
    elif time > 12.0 and time < 16.0:
        return 4
    elif time > 16.0 and time < 20.0:
        return 6
    elif time > 20.0 and time < 24.0:
        return 5
    else:
        return random.randint(1,6)

def direction_5(time):
    if time < 4.0 and time > 0.0:
        return 3
    elif time > 4.0 and time < 8.0:
        return 2
    elif time > 8.0 and time < 12.0:
        return 1
    elif time > 12.0 and time < 16.0:
        return 4
    elif time > 16.0 and time < 20.0:
        return 5
    elif time > 20.0 and time < 24.0:
        return 6
    else:
        return random.randint(1,6)

print(createData(100))