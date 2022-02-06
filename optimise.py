import skopt
import time
import preprocessing
from top_layer import log_results
import model_layer
import numpy as np
from sklearn.svm import SVC
from skopt.callbacks import EarlyStopper


LABEL_TYPE = "timeseries"       # [salad, timeseries]
PREPROCESSING_STEPS = [["pad","BACK",25],["truncate","BACK",25],["transpose"],["flatten"],["onehot-to-labels"]] #["pad","truncate","transpose","flatten","auto_reg_coeff","pca","onehot-to-labels"]
MODEL_TYPES = ["DecidionTree", "RandomForest", "SVC", "KNN", "Bagging"]         # "RandomForest", "SVC", "KNN", "DecidionTree",

PREPROCESSING_STEPS_GENERAL = [
[["pad","BACK",15],["truncate","BACK",15],["transpose"],["flatten"],["onehot-to-labels"]],
[["pad","BACK",20],["truncate","BACK",20],["transpose"],["flatten"],["onehot-to-labels"]],
[["pad","BACK",25],["truncate","BACK",25],["transpose"],["flatten"],["onehot-to-labels"]],

[["pad","FRONT",15],["truncate","FRONT",15],["transpose"],["flatten"],["onehot-to-labels"]],
[["pad","FRONT",20],["truncate","FRONT",20],["transpose"],["flatten"],["onehot-to-labels"]],
[["pad","FRONT",25],["truncate","FRONT",25],["transpose"],["flatten"],["onehot-to-labels"]],

[["transpose"],["auto_reg_coeff"],["flatten"],["onehot-to-labels"]],

[["pad","BACK",15],["truncate","BACK",15],["pca",3],["transpose"],["flatten"],["onehot-to-labels"]],
[["pad","BACK",20],["truncate","BACK",20],["pca",3],["transpose"],["flatten"],["onehot-to-labels"]],
[["pad","BACK",25],["truncate","BACK",25],["pca",3],["transpose"],["flatten"],["onehot-to-labels"]],

[["pad","BACK",15],["truncate","BACK",15],["pca",6],["transpose"],["flatten"],["onehot-to-labels"]],
[["pad","BACK",20],["truncate","BACK",20],["pca",6],["transpose"],["flatten"],["onehot-to-labels"]],
[["pad","BACK",25],["truncate","BACK",25],["pca",6],["transpose"],["flatten"],["onehot-to-labels"]],

[["pad","FRONT",15],["truncate","FRONT",15],["pca",3],["transpose"],["flatten"],["onehot-to-labels"]],
[["pad","FRONT",20],["truncate","FRONT",20],["pca",3],["transpose"],["flatten"],["onehot-to-labels"]],
[["pad","FRONT",25],["truncate","FRONT",25],["pca",3],["transpose"],["flatten"],["onehot-to-labels"]],
                                            
[["pad","FRONT",15],["truncate","FRONT",15],["pca",6],["transpose"],["flatten"],["onehot-to-labels"]],
[["pad","FRONT",20],["truncate","FRONT",20],["pca",6],["transpose"],["flatten"],["onehot-to-labels"]],
[["pad","FRONT",25],["truncate","FRONT",25],["pca",6],["transpose"],["flatten"],["onehot-to-labels"]],

[["pad","BACK",15],["truncate","BACK",15],["transpose"],["pca",6],["flatten"],["onehot-to-labels"]],
[["pad","BACK",20],["truncate","BACK",20],["transpose"],["pca",6],["flatten"],["onehot-to-labels"]],
[["pad","BACK",25],["truncate","BACK",25],["transpose"],["pca",6],["flatten"],["onehot-to-labels"]],
                                                        
[["pad","BACK",15],["truncate","BACK",15],["transpose"],["pca",12],["flatten"],["onehot-to-labels"]],
[["pad","BACK",20],["truncate","BACK",20],["transpose"],["pca",12],["flatten"],["onehot-to-labels"]],
[["pad","BACK",25],["truncate","BACK",25],["transpose"],["pca",12],["flatten"],["onehot-to-labels"]],

[["pad","FRONT",15],["truncate","FRONT",15],["transpose"],["pca",6],["flatten"],["onehot-to-labels"]],
[["pad","FRONT",20],["truncate","FRONT",20],["transpose"],["pca",6],["flatten"],["onehot-to-labels"]],
[["pad","FRONT",25],["truncate","FRONT",25],["transpose"],["pca",6],["flatten"],["onehot-to-labels"]],
                                                          
[["pad","FRONT",15],["truncate","FRONT",15],["transpose"],["pca",12],["flatten"],["onehot-to-labels"]],
[["pad","FRONT",20],["truncate","FRONT",20],["transpose"],["pca",12],["flatten"],["onehot-to-labels"]],
[["pad","FRONT",25],["truncate","FRONT",25],["transpose"],["pca",12],["flatten"],["onehot-to-labels"]]
]

PREPROCESSING_STEPS_ESN = [
[["pad","BACK",15],["truncate","BACK",15],["onehot-to-labels"]],
[["pad","BACK",20],["truncate","BACK",20],["onehot-to-labels"]],
[["pad","BACK",25],["truncate","BACK",25],["onehot-to-labels"]],

[["pad","FRONT",15],["truncate","FRONT",15],["onehot-to-labels"]],
[["pad","FRONT",20],["truncate","FRONT",20],["onehot-to-labels"]],
[["pad","FRONT",25],["truncate","FRONT",25],["onehot-to-labels"]],
]

RF_SPACE = [
    skopt.space.Integer(10, 200, name='n_estimators', prior='uniform'),
    skopt.space.Categorical (["gini", "entropy"], name='criterion')
]
SVC_SPACE = [
    skopt.space.Integer(1, 100000, name = "C", prior='log-uniform'),
    skopt.space.Categorical (["linear","poly","rbf","sigmoid"], name = "kernel"),
    skopt.space.Categorical ([2,3,4], name = "degree"),
    skopt.space.Categorical (["scale","auto"], name = "gamma"),
    skopt.space.Categorical (["ovo","ovr"], name = "decision_function_shape")
]
KNN_SPACE = [
    skopt.space.Integer(1, 20, name='n_neighbors', prior='log-uniform'),
    skopt.space.Categorical (["uniform", "distance"], name='weights'),
    skopt.space.Categorical (["auto","ball_tree","kd_tree","brute"], name = "algorithm"),
    skopt.space.Categorical ([1,2,3,4], name = "p")
]
DTC_SPACE = [
    skopt.space.Categorical (["gini", "entropy"], name='criterion'),
    skopt.space.Categorical (["best", "random"], name='splitter'),
    skopt.space.Integer(1, 12, name='max_depth', prior='uniform')
]
BC_SPACE = [
    skopt.space.Integer(5, 100, name ="n_estimators", prior='uniform'),
    skopt.space.Real(0.01, 1.0, name = "max_samples", prior='uniform'),
    skopt.space.Real(0.01, 1.0, name = "max_features", prior='uniform')
]
RC_SPACE = [
    skopt.space.Integer(1, 20, name = "n_drop", prior='uniform'),
    skopt.space.Categorical([True,False], name ="bidir"),
    skopt.space.Integer(100, 500, name = "n_internal_units", prior='log-uniform'),
    skopt.space.Real(0, 1, name = "spectral_radius", prior='uniform'),
    skopt.space.Real(0, 1, name = "leak", prior='uniform'),
    skopt.space.Categorical([True,False], name ="circle"),
    skopt.space.Real(0, 1, name = "connectivity", prior='uniform'),
    skopt.space.Real(0, 1, name = "input_scaling", prior='uniform'),
    skopt.space.Real(0, 1, name = "noise_level", prior='uniform'),
    skopt.space.Categorical(["tenpca",None], name ="dimred_method"),
    skopt.space.Integer(10,100, name ="n_dim"),
    skopt.space.Categorical(["tenpca",None], name ="dimred_method"),
    skopt.space.Categorical(["last","mean","output","reservoir"], name ="mts_rep"),
    skopt.space.Real(1, 100, name = "w_ridge_embedding", prior='uniform'),
    skopt.space.Categorical(["lin"], name ="readout_type"),
    skopt.space.Real(1, 100, name = "w_ridge", prior='uniform')
]

class StoppingCriterion(EarlyStopper):
    def __init__(self, delta=0.05, n_best=10):
        super(EarlyStopper, self).__init__()
        self.delta = delta
        self.n_best = n_best
    def _criterion(self, result):   
        if len(result.func_vals) >= self.n_best:
            func_vals = np.sort(result.func_vals)
            worst = func_vals[self.n_best - 1]
            best = func_vals[0]
            return abs((best - worst)/worst) < self.delta

def train_evaluate( D, model,params):
    if model == "RandomForest":
        cv_acc = model_layer.random_forest(D, params)[3]
    elif model == "SVC":
        cv_acc = model_layer.support_vector_machine(D, params)[3]
    elif model == "KNN":
        cv_acc = model_layer.KNN(D, params)[3]
    elif model == "Bagging":
        cv_acc = model_layer.bagging_classifier(D, params)[3]
    elif model == "ESN":
        cv_acc = model_layer.echo_state_network(np.asarray(D), params)[3]
    elif model == "DecidionTree":
        cv_acc = model_layer.decidion_tree(D, params)[3]
    return cv_acc

if __name__ == '__main__':
    start_time = time.time()
    print("Starting..")

    @skopt.utils.use_named_args(RF_SPACE)
    def objectiveRF(**params):
        all_params = {**params}
        return -1.0 * train_evaluate(D, "RandomForest", all_params)

    @skopt.utils.use_named_args(SVC_SPACE)
    def objectiveSVC(**params):
        all_params = {**params}
        return -1.0 * train_evaluate(D, "SVC", all_params)

    @skopt.utils.use_named_args(KNN_SPACE)
    def objectiveKNN(**params):
        all_params = {**params}
        return -1.0 * train_evaluate(D, "KNN", all_params)

    @skopt.utils.use_named_args(BC_SPACE)
    def objectiveBC(**params):
        all_params = {**params}
        return -1.0 * train_evaluate(D, "Bagging", all_params)
    
    @skopt.utils.use_named_args(RC_SPACE)
    def objectiveRC(**params):
        all_params = {**params}
        return -1.0 * train_evaluate(D, "ESN", all_params)

    @skopt.utils.use_named_args(DTC_SPACE)
    def objectiveDTC(**params):
        all_params = {**params}
        return -1.0 * train_evaluate(D, "DecidionTree", all_params)

    for preprocess in PREPROCESSING_STEPS_GENERAL:
        D = preprocessing.get_data(label_type=LABEL_TYPE, preproccessing_steps=preprocess)
        D_name = ' '.join(map(str,preprocess))
        for model in MODEL_TYPES:
            if model == "RandomForest":
                try:
                    results = skopt.gbrt_minimize(objectiveRF, RF_SPACE, n_calls=100, n_jobs=3, callback = skopt.callbacks.CheckpointSaver('opt/RF/'+D_name+'.pkl', store_objective=False))
                    skopt.dump(results, 'opt/RF/'+D_name+'.pkl', store_objective=False)
                except:
                    pass
            elif model == "SVC":
                try:
                    results = skopt.gbrt_minimize(objectiveSVC, SVC_SPACE, n_calls=100, n_jobs=3, callback = skopt.callbacks.CheckpointSaver('opt/SVC/'+D_name+'.pkl', store_objective=False))
                    skopt.dump(results, 'opt/SVC/'+D_name+'.pkl', store_objective=False)
                except:
                    pass
            elif model == "KNN":
                try:
                    results = skopt.gbrt_minimize(objectiveKNN, KNN_SPACE, n_calls=100, n_jobs=3, callback = skopt.callbacks.CheckpointSaver('opt/KNN/'+D_name+'.pkl', store_objective=False))
                    skopt.dump(results, 'opt/KNN/'+D_name+'.pkl', store_objective=False)
                except:
                    pass
            elif model == "Bagging":
                try:
                    results = skopt.gbrt_minimize(objectiveBC, BC_SPACE, n_calls=100, n_jobs=3, callback = skopt.callbacks.CheckpointSaver('opt/BC/'+D_name+'.pkl', store_objective=False))
                    skopt.dump(results, 'opt/BC/'+D_name+'.pkl', store_objective=False)
                except:
                    pass
            elif model == "DecidionTree":
                try:
                    results = skopt.gbrt_minimize(objectiveDTC, DTC_SPACE, n_calls=100, n_jobs=3, callback = skopt.callbacks.CheckpointSaver('opt/DTC/'+D_name+'.pkl', store_objective=False))
                    skopt.dump(results, 'opt/DTC/'+D_name+'.pkl', store_objective=False)
                except:
                    pass





    #for preprocess in PREPROCESSING_STEPS_ESN:
    #    D = preprocessing.get_data(label_type=LABEL_TYPE, preproccessing_steps=preprocess)
    #    D_name = ' '.join(map(str,preprocess))
    
    #    results = skopt.gbrt_minimize(objectiveRF, RC_SPACE, n_calls=100, n_jobs=3)
    #    skopt.dump(results, 'opt/ESN/'+D_name+'.pkl', store_objective=False)
                

    end_time = time.time()
    print(f'\nElapsed time: {end_time - start_time}')
    