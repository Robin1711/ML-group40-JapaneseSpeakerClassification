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
        else:
            labels = np.zeros(9)
            labels[speakerIndex] = 1

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
        else:
            labels = np.zeros(9)    
            labels[speakerIndex] = 1
        y.append(labels)
    return y

## this function does what the provided matlab dataloader does: one hot encoding for each timestep of the utterance
def load_w_label_timestep():
    raw_train = read_raw_file_lines(train_data_path)
    raw_test = read_raw_file_lines(test_data_path)

    train_x = np.asarray(proccess_x(raw_train),dtype=object)
    test_x = np.asarray(proccess_x(raw_test),dtype=object)
    
    train_y = np.asarray(proccess_train_y(train_x, timestep=True),dtype=object)
    test_y = np.asarray(proccess_test_y(test_x, timestep=True),dtype=object)

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
    parser.add_argument('-labels_per_timestep', type=bool, default=False, help='if added each time step of the utterance has a label, otherwise the label is associated with the entire timeserise of the utterance.')

    args = parser.parse_args()
    print(args.labels_per_timestep)
    if args.labels_per_timestep:
        print("Loading data with timestep labels")
        load_w_label_timestep()
    else:
        print("Loading data with timeseries labels")
        load_w_label_timeseries()
    