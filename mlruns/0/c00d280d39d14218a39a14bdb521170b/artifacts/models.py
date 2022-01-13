import mlflow
from sklearn import metrics as sk_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
import preprocessing


def randomForest(n_trees, criterion):
    ################################
    params = {
    "classifier" : "random forest",
    "n_trees" : n_trees,
    "criterion" : criterion
    }
    metrics = {
    }
    ################################

    model = RandomForestClassifier(n_estimators=n_trees, criterion=criterion)

    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    print(scores)

    y_pred = model.predict(np.asarray(X_test))

    
    metrics["accuracy"] = sk_metrics.accuracy_score(y_test, y_pred)
    metrics["fScore"] = sk_metrics.f1_score(y_test, y_pred, average='weighted')
    metrics["precision"] = sk_metrics.precision_score(y_test, y_pred, average='weighted')

    
    print("%s: [Accuracy: %f, F-Score: %f, precision: %f]" % (params['classifier'],metrics["accuracy"],metrics["fScore"],metrics["precision"],))

    return params, metrics

if __name__ == '__main__':
    label_type= "salad"
    preproccessing_steps = []
    D = preprocessing.get_data(label_type, preproccessing_steps)

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
    
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("./models.py")

    