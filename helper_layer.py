from itertools import chain

import mlflow
import numpy as np
import time
import random

from sklearn import metrics as sk_metrics
from sklearn.model_selection import permutation_test_score, StratifiedShuffleSplit, cross_val_score

# Random Seed
random.seed(2022)

N_CPUS = 3                      # -1 denotes all cpu's
CROSS_VALIDATION_NSPLITS = 5

# Not necessary?
# def crossValidation(model, name:str, k:int, label_type:str = "salad", preproccessing_steps:list = [], test_size = None, train_size = None):
#     ################################
#     params = {
#     "classifier" : "random forest",
#     "n data splits" : k,
#     "test size" : test_size,
#     "train size" : train_size,
#     "label type" : label_type,
#     "preproccessing steps": preproccessing_steps,
#
#     }
#     metrics = {
#     }
#     ################################
#     # get a dataset with no preproccessing steps applied so there is no data leaking in the cross validation
#     D = preprocessing.get_data(label_type)
#     X_train = D[0]
#     y_train = D[1]
#     #X_test = D[2]
#     #y_test = D[3]
#
#     sss = StratifiedShuffleSplit(n_splits=k, test_size=test_size, train_size=train_size)
#
#     ## corss validation loop
#     for train_index, test_index in sss.split(X_train, y_train):
#         D1 = [X_train[train_index], y_train[train_index], X_train[test_index], y_train[test_index]]
#
#         ## individual preprocessing for each set
#         for i,dataset in enumerate(D1):
#             D1[i] = preprocessing.get_data(dataset, label_type, preproccessing_steps)
#
#         X_val_train = D[0]
#         y_val_train = D[1]
#         #X_val_test = D[2]
#         #y_val_test = D[3]
#
#         ## whateveer cross validation stuff we wanna do
#         #score,_,pvalue = permutation_test_score(model,X_val_train,y_val_train,scoring='accuracy',cv=10,n_jobs=-1)
#         #print("permutation test: [accruacy: %f, pvalue: %f]" % (score,pvalue))
#         #scores = cross_val_score(model,X_train,y_train,scoring="accuracy",cv=10,n_jobs=-1)
#         #print("cross val: [accruacy: %f]" % (np.mean(scores)))

def evaluate_model(x_train, y_train, x_test, y_test, model):
    metrics = dict()
    y_train_pred = model.predict(x_train)
    metrics["accuracy_train"] = sk_metrics.accuracy_score(y_train, y_train_pred)

    y_test_pred = model.predict(x_test)
    metrics["accuracy_test"] = sk_metrics.accuracy_score(y_test, y_test_pred)
    metrics["fScore_test"] = sk_metrics.f1_score(y_test, y_test_pred, average='weighted')
    metrics["precision_test"] = sk_metrics.precision_score(y_test, y_test_pred, average='weighted')
    metrics["confusion_matrix"] = sk_metrics.confusion_matrix(y_test, y_test_pred, labels=list(range(0, 9)))
    return metrics


def model_cv_train(x_train, y_train, model=None):
    print(f"Model = {model}")

    print("Permutation Test Model..")
    permutation_start_time = time.time()
    score, _, pvalue = permutation_test_score(model, x_train, y_train
                                              , scoring='accuracy'
                                              , cv=StratifiedShuffleSplit(n_splits=CROSS_VALIDATION_NSPLITS)
                                              , n_jobs=N_CPUS
                                              , verbose=1)
    print(f"Permutation Test Elapsed time: {time.time() - permutation_start_time}")
    print(f"Permutation Test: [accuracy: {score}, pvalue: {pvalue}]")

    print("Cross Validating Model..")
    cv_start_time = time.time()
    scores = cross_val_score(model, x_train, y_train
                             , scoring="accuracy"
                             , cv=StratifiedShuffleSplit(n_splits=CROSS_VALIDATION_NSPLITS)
                             , n_jobs=N_CPUS
                             , verbose=2)
    print(f"Cross Validation Elapsed time: {time.time() - cv_start_time}")
    print(f"Cross Validation: [accuracy: {np.average(scores)}]")

    model.fit(x_train, y_train)
    return model
