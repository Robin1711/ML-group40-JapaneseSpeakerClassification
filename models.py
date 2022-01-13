import preprocessing
import argparse
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn import metrics
import numpy as np

#GMLVQ library
import sklvq
from sklvq import GMLVQ


#K Means Classifier
def kmeans(X_train, y_train):
    kmeans = KMeans(n_clusters=9, random_state=0).fit(X)
    kmeans.labels_
    kmeans.predict([[0, 0], [12, 3]])

def knn(X_train, y_train):
    pass

#Random Forest Classifier
def randomForest(D, n_trees, split):
    X_train = D[0]
    y_train = D[1]
    X_test = D[2]
    y_test = D[3]

    model = RandomForestClassifier(n_estimators=n_trees, criterion=split)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    #fScore = metrics.f1_score(y_test, y_pred, )

    return [accuracy]

def run_RandomForest():
    D = preprocessing.get_data(label_type= "salad", preproccessing_steps = [])

    ## model parameters:
    n_trees = 100
    criterion = 'gini'
    accuracy = randomForest(D, n_trees,criterion)

    print(accuracy)
    #print(fScore)

    #with MLflow record the parameretes and resutls

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
        }
        random_state=1428
    )

    model.fit(train_data, train_labels)

def run_gmlvq(test_data, test_labels):
    predicted_labels = model.predict(test_data)
    print("GMLVQ:\n" + classification_report(test_labels, predicted_labels))
    return(predicted_labels)

if __name__ == '__main__':
    run_RandomForest()
