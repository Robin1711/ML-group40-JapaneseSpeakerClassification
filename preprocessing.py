from statsmodels.tsa.ar_model import AutoReg
import data_loader
import argparse
import numpy as np
import data_formatting
from itertools import chain
from RC.tensorPCA import tensorPCA



def auto_reg_coeff(data, lag):
    for i, recording in enumerate(data):
        for j, feature in enumerate(recording):
            ar_model = AutoReg(data[i][j], lags=0).fit()
            data[i][j] = ar_model.params
    
    return data

def pca(data):
    pca_model = tensorPCA(n_components = 10)
    return pca_model.fit_transform(np.asarray(data))

def get_data(D = None, label_type:str = "timestep", preproccessing_steps:list = [], signal_length:int = 22, ) -> list:
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
        if step == "pad":
            D[0] = data_formatting.pad_signals(D[0], padding_type="BACK", padding_to=signal_length)
            D[2] = data_formatting.pad_signals(D[2], padding_type="BACK", padding_to=signal_length)
        elif step == "truncate":
            D[0] = data_formatting.truncate_signals(D[0], truncation_type="BACK", truncate_to=signal_length)
            D[2] = data_formatting.truncate_signals(D[2], truncation_type="BACK", truncate_to=signal_length)
        elif step == "transpose":
            D[0] = data_formatting.transpose(D[0])
            D[2] = data_formatting.transpose(D[2])
        elif step == "auto_reg_coeff":
            D[0] = auto_reg_coeff(D[0], lag=0)
            D[2] = auto_reg_coeff(D[2], lag=0)
        elif step == "flatten":
            D[0] = [list(chain(*example)) for example in D[0]]
            D[2] = [list(chain(*example)) for example in D[2]]
        elif step == "pca":
            D[0] = pca(D[0])
            D[2] = pca(D[2])
        elif step == "onehot-to-labels":
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
