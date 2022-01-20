import mlflow
from sklearn import metrics as sk_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

import numpy as np
import preprocessing

#GMLVQ library
import sklvq
from sklvq import GMLVQ


#K Means Classifier
def kmeans(X_train, y_train):
    kmeans = KMeans(n_clusters=9, random_state=0).fit(X)
    kmeans.labels_
    kmeans.predict([[0, 0], [12, 3]])
#
def crossValidation(model, name:str, k:int, label_type:str = "salad", preproccessing_steps:list = [], test_size = None, train_size = None):
    ################################
    params = {
    "classifier" : "random forest",
    "n data splits" : k,
    "test size" : test_size,
    "train size" : train_size,
    "label type" : label_type,
    "preproccessing steps": preproccessing_steps,

    }
    metrics = {
    }
    ################################
    # get a dataset with no preproccessing steps applied so there is no data leaking in the cross validation
    D = preprocessing.get_data(label_type)
    X_train = D[0]
    y_train = D[1]
    #X_test = D[2]
    #y_test = D[3]

    sss = StratifiedShuffleSplit(n_splits=k, test_size=test_size, train_size=train_size)

    ## corss validation loop
    for train_index, test_index in sss.split(X_train, y_train):
        D1 = [X_train[train_index], y_train[train_index], X_train[test_index], y_train[test_index]]

        ## individual preprocessing for each set
        for i,dataset in enumerate(D1):
            D1[i] = preprocessing.get_data(dataset, label_type, preproccessing_steps)

        X_val_train = D[0]
        y_val_train = D[1]
        #X_val_test = D[2]
        #y_val_test = D[3]

        ## whateveer cross validation stuff we wanna do
        #score,_,pvalue = permutation_test_score(model,X_val_train,y_val_train,scoring='accuracy',cv=10,n_jobs=-1)
        #print("permutation test: [accruacy: %f, pvalue: %f]" % (score,pvalue))
        #scores = cross_val_score(model,X_train,y_train,scoring="accuracy",cv=10,n_jobs=-1)
        #print("cross val: [accruacy: %f]" % (np.mean(scores)))


def randomForest(n_trees, criterion):
    ################################
    params = {
    "classifier" : "random forest",
    "n_trees" : n_trees,
    "split criterion" : criterion
    }
    metrics = {
    }
    ################################

    model = RandomForestClassifier(n_estimators=n_trees, criterion=criterion)

    #score,_,pvalue = permutation_test_score(model,X_train,y_train,scoring='accuracy',cv=StratifiedShuffleSplit(),n_jobs=-1)
    #print("permutation test: [accruacy: %f, pvalue: %f]" % (score,pvalue))
    #scores = cross_val_score(model,X_train,y_train,scoring="accuracy",cv=StratifiedShuffleSplit(),n_jobs=-1)
    #print("cross val: [accruacy: %f]" % (np.mean(scores)))

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)


    metrics["accuracy"] = sk_metrics.accuracy_score(y_test, y_pred)
    metrics["fScore"] = sk_metrics.f1_score(y_test, y_pred, average='weighted')
    metrics["precision"] = sk_metrics.precision_score(y_test, y_pred, average='weighted')


    print("%s: [Accuracy: %f, F-Score: %f, precision: %f]" % (params['classifier'],metrics["accuracy"],metrics["fScore"],metrics["precision"],))

    return params, metrics

def gmlvq(train_data, train_labels):
    #temporarily hardcoded values
    max_runs = 100
    k = 2
    step_size = np.array([0.1, 0.05])

    model = GMLVQ(
        distance_type="adaptive-squared-euclidian",
        activation_type="identity",
        solver_type="waypoint-gradient-descent",
        solver_params={
            "max_runs": max_runs,
            "k": k,
            "step_size": step_size
        },
        random_state=1428
    )

    model.fit(train_data, train_labels)
    return(model)

def run_gmlvq(test_data, test_labels, model):
    predicted_labels = model.predict(test_data)
    print("GMLVQ:\n" + classification_report(test_labels, predicted_labels))
    return(predicted_labels)

if __name__ == '__main__':
    label_type= "salad"
    preproccessing_steps = []
    D = preprocessing.get_data(label_type=label_type, preproccessing_steps=preproccessing_steps)

    global  X_train
    global  y_train
    global  X_test
    global  y_test

    X_train = D[0]
    y_train = D[1]
    X_test = D[2]
    y_test = D[3]

    params, metrics = randomForest(n_trees=100, criterion="gini")
    params["label type"] = label_type
    params["preproccessing steps"] = preproccessing_steps

    mlflow.log_params(params) # log parameters
    mlflow.log_metrics(metrics) # log metrics
    mlflow.log_artifact("./models.py") # save a copy of models
