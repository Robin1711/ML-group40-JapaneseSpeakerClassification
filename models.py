import preprocessing
import argparse
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np




def kmeans(X_train, y_train):
    kmeans = KMeans(n_clusters=9, random_state=0).fit(X)
    kmeans.labels_
    kmeans.predict([[0, 0], [12, 3]])

def knn(X_train, y_train):
    pass

def randomForest(D, n_trees, split):
    X_train = np.asarray(D[0])
    y_train = np.asarray(D[1])
    X_test = np.asarray(D[2])
    y_test = np.asarray(D[3])

    model = RandomForestClassifier(n_estimators=n_trees, criterion=split)
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    fScore = metrics.f1_score(y_test, y_pred)

    return [accuracy, fScore]

def run_RandomForest():
    D = preprocessing.get_data(label_type= "timeseries", preproccessing_steps = [])

    ## model parameters:
    n_trees = 100
    criterion = 'gini'
    accuracy, fScore = randomForest(D, n_trees,criterion)

    print(accuracy)
    print(fScore)

    #with MLflow record the parameretes and resutls

if __name__ == '__main__':
    run_RandomForest()

    