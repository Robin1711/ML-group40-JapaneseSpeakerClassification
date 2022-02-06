from statsmodels.tsa.ar_model import AutoReg
import data_loader
import argparse
import numpy as np
import data_formatting
from itertools import chain
from RC.tensorPCA import tensorPCA



def auto_reg_coeff(data):
    new_rep = np.zeros((len(data),len(data[0]),1))
    for i, recording in enumerate(data):
        for j, feature in enumerate(recording):
            ar_model = AutoReg(data[i][j], lags=0).fit()
            new_rep[i][j] = ar_model.params
    
    return new_rep

def pca(data, dimensions):
    pca_model = tensorPCA(n_components = dimensions)
    return pca_model.fit_transform(np.asarray(data))

def get_data(D = None, label_type:str = "timestep", preproccessing_steps:list = []) -> list:
    """Args:
        D: a specific data set to avoid possible data leakage during cross validation
        label_type: timestep/timeseries/salad
    """
    if D == None:
        if label_type == "timestep":
            D = data_loader.load_w_label_timestep()
        elif label_type == "timeseries":
            D = data_loader.load_w_label_timeseries()
        elif label_type == "salad":
            D = data_loader.load_w_label_salad()
        else:
            print("label type not implemented")
            exit()
    
    for step in preproccessing_steps:
        if step[0] == "pad":
            D[0] = data_formatting.pad_signals(D[0], padding_type=step[1], padding_to=step[2])
            D[2] = data_formatting.pad_signals(D[2], padding_type=step[1], padding_to=step[2])
        elif step[0] == "truncate":
            D[0] = data_formatting.truncate_signals(D[0], truncation_type=step[1], truncate_to=step[2])
            D[2] = data_formatting.truncate_signals(D[2], truncation_type=step[1], truncate_to=step[2])
        elif step[0] == "transpose":
            D[0] = data_formatting.transpose(D[0])
            D[2] = data_formatting.transpose(D[2])
        elif step[0] == "auto_reg_coeff":
            D[0] = auto_reg_coeff(D[0])
            D[2] = auto_reg_coeff(D[2])
        elif step[0] == "flatten":
            D[0] = [list(chain(*example)) for example in D[0]]
            D[2] = [list(chain(*example)) for example in D[2]]
        elif step[0] == "pca":
            D[0] = pca(D[0], step[1])
            D[2] = pca(D[2], step[1])
        elif step[0] == "onehot-to-labels":
            D[1] = data_formatting.transform_labelvectors_to_labels(D[1])
            D[3] = data_formatting.transform_labelvectors_to_labels(D[3])
        else:
           print("preproccessing step: " + step + " not implemented")
           exit()

    return D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load train/test data in a specific format')
    #parser.add_argument('-labels_per_timestep', type=bool, default=False, help='if added each time step of the utterance has a label, otherwise the label is associated with the entire timeserise of the utterance.')

    args = parser.parse_args()
