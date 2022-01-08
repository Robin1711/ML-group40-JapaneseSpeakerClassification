import data_loader
import argparse

def PCA():
    pass

def LDA():
    pass

def tSNE():
    pass

def get_data(label_type:str = "timestep", preproccessing_steps:list = []) -> list:
    """Args:
        label_type: 
    """
    if label_type == "timestep":
        D = data_loader.load_w_label_timestep()
    elif label_type == "timeseries":
        D = data_loader.load_w_label_timeseries()
    else:
        print("label type not implemented")
        exit()
    
    for step in preproccessing_steps:
        #if step == "pca":
        #    D = PCA(D)
        #elif step == "lda":
        #    D = LDA(D)
        #elif step == "tsne":
        #    D = tSNE(D)
        #else:
        #   print("preproccessing step: " + step + " not implemented")
        #   exit()
        pass
    
    return D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load train/test data in a specific format')
    #parser.add_argument('-labels_per_timestep', type=bool, default=False, help='if added each time step of the utterance has a label, otherwise the label is associated with the entire timeserise of the utterance.')

    args = parser.parse_args()

    