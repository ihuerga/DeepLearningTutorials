__author__ = 'huergasi'


from os import listdir
from os.path import isfile, join
import numpy as np
import theano as t

def parsefile(filename, targetoffset):
    indices = [1,2,3,4,6,7,8,9,11,12,13,14]
    volume_index = [16]
    original_data = getoriginaldata(filename)
    file = open(filename, 'r')
    start = 0
    previous = []
    x=[]
    y=[]
    i = 0
    for line in file:
        row = np.array(line.split(','))
        current = np.take(row, indices).astype(np.float32)
        if start == 0:
            previous = np.take(row, indices).astype(np.float32)
            previous_volume = np.take(row, volume_index).astype(np.float32)
            start+=1
        else:
            current_volume = np.take(row, volume_index).astype(np.float32)
            sample = np.empty((1,current.size + 1))
            for index in np.arange(current.size):
                sample[0][index] = ((current[index] - previous[index]) / previous[index]) * 100
            sample[0][current.size] = current_volume
            previous = current
            if i  < (original_data.size / 13) - targetoffset:
                x .append(sample)
                current_y = ((original_data[(i + targetoffset)*13] - current[0]) / current[0]) * 100
                y.append(current_y)
            i+=1
    return np.array(x),np.array(y)

def parse_and_normalize_file(filename, targetoffset, mu, sigma, y_max, y_min):
    indices = [1,2,3,4,6,7,8,9,11,12,13,14]
    volume_index = [16]
    original_data = getoriginaldata(filename)
    file = open(filename, 'r')
    start = 0
    previous = []
    x= np.empty([1,13], dtype=np.float32)
    y= np.empty([1], dtype=np.float32)
    i = 0
    for line in file:
        row = np.array(line.split(','))
        current = np.take(row, indices).astype(np.float32)
        if start == 0:
            previous = np.take(row, indices).astype(np.float32)
            previous_volume = np.take(row, volume_index).astype(np.float32)
            start+=1
        else:
            current_volume = np.take(row, volume_index).astype(np.float32)
            sample = np.empty((1,current.size + 1))
            for index in np.arange(current.size):
                sample[0][index] = ((current[index] - previous[index]) / previous[index]) * 100
            sample[0][current.size] = current_volume
            previous = current
            if i  < (original_data.size / 13) - targetoffset:
                sample = ((sample - mu )/ sigma).astype(np.float32)
                x = np.vstack((x, sample))
                current_y = (((original_data[(i + targetoffset)*13] - current[0]) / current[0]) * 100).astype(np.float32)
                current_y = ((2 * (current_y - y_min) / (y_max-y_min)) - 1).astype(np.float32)
                y = np.vstack((y,current_y))
            i+=1
    x = np.delete(x, 0, 0)
    y = np.delete(y, 0, 0)
    return x, y


def get_mu_sigma_y(filepath, targetoffset):
    files = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    alldata = []
    ally=[]
    mu = np.zeros((13,), dtype=np.float32)
    sigma = np.zeros((13,), dtype=np.float32)
    y_max = 0
    y_min = 0
    for f in files:
        x,y = (parsefile(join(filepath, f),targetoffset))
        alldata.append(x)
        ally.append(y)

    alldata_matrix = alldata[0]
    ally_matrix = ally[0]

    start = 0
    for el in alldata:
        if start !=0:
            alldata_matrix=np.vstack((alldata_matrix, el))
        else:
            start+=1

    start = 0
    for ey in ally:
        if start !=0:
            ally_matrix=np.hstack((ally_matrix, ey))
        else:
            start+=1

    alldata_matrix=np.matrix(alldata_matrix)
    mu = alldata_matrix.mean(0)
    sigma = alldata_matrix.std(0)
    y_max = ally_matrix.max()
    y_min = ally_matrix.min()

    return mu, sigma, y_max, y_min

def getoriginaldata(filename):
    indices = [1,2,3,4,6,7,8,9,11,12,13,14,16]
    file = open(filename, 'r')
    data = []
    start = 0
    for line in file:
        if start>0:
            row = np.array(line.split(','))
            data = np.append(data, np.take(row, indices).astype(np.float))
        start+=1
    return data

def getdata():
    path_training=r"C:\Users\huergasi\MLCode\myCode\data\ib\30sec\training_sample"
    path_testing=r"C:\Users\huergasi\MLCode\myCode\data\ib\30sec\testing_sample"
    path_validation=r"C:\Users\huergasi\MLCode\myCode\data\ib\30sec\validation_sample"

    training_files= [f for f in listdir(path_training) if isfile(join(path_training, f))]
    testing_files= [f for f in listdir(path_testing) if isfile(join(path_testing, f))]
    validation_files=[f for f in listdir(path_validation) if isfile(join(path_validation, f))]

    training_data_x = []
    training_data_y = []

    testing_data_x = []
    testing_data_y = []

    validation_data_x = []
    validation_data_y = []

    targetoffset = 4

    mu, sigma, y_max, y_min = get_mu_sigma_y(path_training, targetoffset)

    for f in training_files:
        x_train,y_train = parse_and_normalize_file(join(path_training, f),targetoffset, mu, sigma, y_max, y_min)
        training_data_x.append(x_train)
        training_data_y.append(y_train)

    for ftest in testing_files:
        x_test,y_test = parse_and_normalize_file(join(path_testing, ftest),targetoffset, mu, sigma, y_max, y_min)
        testing_data_x.append(x_test)
        testing_data_y.append(y_test)

    for fval in validation_files:
        x_val,y_val = parse_and_normalize_file(join(path_validation, fval),targetoffset, mu, sigma, y_max, y_min)
        validation_data_x.append(x_val)
        validation_data_y.append(y_val)

    return (training_data_x, training_data_y), (validation_data_x, validation_data_y), (testing_data_x, testing_data_y)





