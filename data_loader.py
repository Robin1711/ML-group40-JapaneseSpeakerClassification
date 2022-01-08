from sys import implementation
import numpy as np
from matplotlib import pyplot as plt
import argparse

train_data_path = './data/ae.train'
test_data_path = './data/ae.test'

def read_raw_file_lines(filename):
    with open(filename) as f:
        contents = f.readlines()
    return contents

def proccess_x(dataset):
    x = []
    start_idx = 0
    for idx, line in enumerate(dataset):
        if line[0:4] != '1.0 ':
            continue
        else:
            utterance_str = dataset[start_idx:idx]
            utterance = []
            for str_line in utterance_str:
                feature = [float(num) for num in str_line.strip().split(" ") if num != ""]
                if feature != []:
                    utterance.append(feature)
            x.append(utterance)
            start_idx = idx + 1
    return x

def proccess_train_y(dataset, timestep = True):
    y = []
    for i in range(len(dataset)):
        speakerIndex = int(np.floor(i/30))

        if timestep == True:
            l = len(dataset[i])
            labels = np.zeros((l,9))
            labels[:,speakerIndex] = 1
            labels = labels.tolist()
        else:
            labels = list(np.zeros(9))
            labels[speakerIndex] = 1
            labels = labels.tolist()


        y.append(labels)
    return y

def proccess_test_y(dataset, timestep = True):
    y = []
    speakerIndex = 0
    blockCounter = 0
    blockLengthes = [31, 35, 88, 44, 29, 24, 40, 50, 29]
    for i in range(len(dataset)):
        blockCounter += 1
        if blockCounter == blockLengthes[speakerIndex] + 1:
            speakerIndex += 1
            blockCounter = 1
        if timestep == True:
            l = len(dataset[i])
            labels = np.zeros((l,9))   
            labels[:,speakerIndex] = 1
            labels = labels.tolist()
        else:
            labels = list(np.zeros(9))
            labels[speakerIndex] = 1
            labels = labels.tolist()

        y.append(labels)
    return y

## "salad-ification" remove the data structure that keeps the data organised by utterance so one utterance will be inditinguishable from another
def saladify(data):
    salad = []
    for utterance in data:
        for timestep in utterance:
            salad.append(timestep)
    return salad

## this function does what the provided matlab dataloader does: one hot encoding for each timestep of the utterance
def load_w_label_timestep():
    raw_train = read_raw_file_lines(train_data_path)
    raw_test = read_raw_file_lines(test_data_path)

    train_x = proccess_x(raw_train)
    test_x = proccess_x(raw_test)
    
    train_y = proccess_train_y(train_x, timestep=True)
    test_y = proccess_test_y(test_x, timestep=True)

    return [train_x, train_y, test_x, test_y]

# loading with this removes all timesereies information acciated with the data, generally this is not a good idea to use for the classifier but good for easy preliminary testing
def load_w_label_salad():
    D = load_w_label_timestep()
    train_x = D[0]
    train_y = D[1]
    test_x = D[2]
    test_y = D[3]

    train_x = saladify(train_x)
    train_y = saladify(train_y)
    test_x = saladify(test_x)
    test_y = saladify(test_y)
    x=1
    return [train_x, train_y, test_x, test_y]


## this function returns the data with labels for each time searies one hot encoding.
def load_w_label_timeseries():
    raw_train = read_raw_file_lines(train_data_path)
    raw_test = read_raw_file_lines(test_data_path)

    train_x = np.asarray(proccess_x(raw_train),dtype=object)
    test_x = np.asarray(proccess_x(raw_test),dtype=object)
    
    train_y = np.asarray(proccess_train_y(train_x, timestep=False),dtype=object)
    test_y = np.asarray(proccess_test_y(test_x, timestep=False),dtype=object)

    return [train_x, train_y, test_x, test_y]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load train/test data in a specific format')
    parser.add_argument('--label_type', type=str, help='timestep/timeserise/salad')

    args = parser.parse_args()
    print(args.label_type)
    if args.label_type == 'timestep':
        print("Loading data with timestep labels")
        load_w_label_timestep()
    elif args.label_type == 'timeserise':
        print("Loading data with timeseries labels")
        load_w_label_timeseries()
    elif args.label_type == 'salad':
        print("Loading data with salad labels")
        load_w_label_salad()
    else:
        print("loading format has not implemented")
    