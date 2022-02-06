from itertools import chain

import mlflow
import numpy as np
import time
import random

from sklearn import metrics as sk_metrics
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import KFold, permutation_test_score, StratifiedShuffleSplit, cross_val_score, ShuffleSplit
import warnings
warnings.filterwarnings("error")

# Random Seed
random.seed(2022)

N_CPUS = 3                      # -1 denotes all cpu's
CROSS_VALIDATION_NSPLITS = 10

def evaluate_model(x_train, y_train, x_test, y_test, model):
    metrics = dict()
    y_train_pred = model.predict(np.asarray(x_train))
    metrics["accuracy_train"] = sk_metrics.accuracy_score(y_train, y_train_pred)

    y_test_pred = model.predict(np.asarray(x_test))
    metrics["accuracy_test"] = sk_metrics.accuracy_score(y_test, y_test_pred)
    metrics["fScore_test"] = sk_metrics.f1_score(y_test, y_test_pred, average='weighted')
    metrics["precision_test"] = sk_metrics.precision_score(y_test, y_test_pred, average='weighted')
    metrics["confusion_matrix"] = sk_metrics.confusion_matrix(y_test, y_test_pred, labels=list(range(0, 9)))
    return metrics


def model_cv_train(x_train, y_train, model=None, trained=False):
    CV = StratifiedShuffleSplit(n_splits=CROSS_VALIDATION_NSPLITS)
    print(f"Model = {model}")

    print("Cross Validating Model..")
    cv_start_time = time.time()
    try:
        scores = cross_val_score(model, x_train, y_train
                             , scoring="accuracy"
                             , cv=CV
                             , n_jobs=N_CPUS
                             , verbose=2)
    except:
        model = DummyClassifier(strategy="most_frequent")
        scores = [0.5]
    print(f"Cross Validation Elapsed time: {time.time() - cv_start_time}")
    print(f"Cross Validation: [accuracy: {np.average(scores)}]")


    model.fit(x_train, y_train)
    
    return model, np.average(scores)
