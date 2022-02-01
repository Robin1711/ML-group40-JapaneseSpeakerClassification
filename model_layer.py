import numpy as np

# Classifiers
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# GMLVQ libraries
import sklvq
from sklvq import GMLVQ

# Self defined modules
import helper_layer

PARAMS_RF = {"classifier": "Random Forest Classifier", "n_trees": 25, "split_criterion": "gini"}
PARAMS_SVC = {"classifier": "Support Vector Classifier", "C": 1.0, "kernel": "rbf"}
PARAMS_ENSEMBLE = {"classifier": "Ensemble Voting Classifier", "voting": 'hard'}
PARAMS_KNN = {"classifier": "K Nearest Neighbors", "n_neighbors": 7, "cpus": 3}
PARAMS_DTC = {"classifier": "Decision Tree", "max_depth": 8}
PARAMS_BC = {"classifier": "Bagging Classifier - SVC", "n_estimators": 5, "max_samples": 0.8, "max_features": 1.0}


def random_forest(data):
    x_train, y_train, x_test, y_test = data
    model = RandomForestClassifier(
        n_estimators=PARAMS_RF["n_trees"]
        , criterion=PARAMS_RF["split_criterion"]
        , random_state=0
    )

    trained_model = helper_layer.model_cv_train(x_train, y_train, model)
    metrics = helper_layer.evaluate_model(x_train, y_train, x_test, y_test, trained_model)
    return PARAMS_RF, metrics, trained_model


def support_vector_machine(data):
    x_train, y_train, x_test, y_test = data
    model = SVC(
        C=PARAMS_SVC["C"]
        , kernel=PARAMS_SVC["kernel"]
        , random_state=0
    )

    trained_model = helper_layer.model_cv_train(x_train, y_train, model)
    metrics = helper_layer.evaluate_model(x_train, y_train, x_test, y_test, trained_model)
    return PARAMS_SVC, metrics, trained_model


def ensemble_classifier(data):
    x_train, y_train, x_test, y_test = data

    classifiers = [
        KNeighborsClassifier(n_neighbors=PARAMS_KNN["n_neighbors"], n_jobs=PARAMS_KNN["cpus"]),
        SVC(C=PARAMS_SVC["C"], kernel=PARAMS_SVC["kernel"], probability=(PARAMS_ENSEMBLE["voting"] == "soft"),
            random_state=0),
        DecisionTreeClassifier(max_depth=PARAMS_DTC["max_depth"], random_state=0)
    ]
    ensemble_model = VotingClassifier(
        [(e.__class__.__name__, e) for e in classifiers]
        , voting=PARAMS_ENSEMBLE["voting"]
    )

    trained_model = helper_layer.model_cv_train(x_train, y_train, ensemble_model)
    metrics = helper_layer.evaluate_model(x_train, y_train, x_test, y_test, trained_model)
    return PARAMS_ENSEMBLE, metrics, trained_model


def bagging_classifier(data):
    x_train, y_train, x_test, y_test = data

    bagging_model = BaggingClassifier(
        base_estimator=SVC(
            C=PARAMS_SVC["C"]
            , kernel=PARAMS_SVC["kernel"]
            , random_state=0)
        , n_estimators=PARAMS_BC["n_estimators"]
        , max_samples=PARAMS_BC["max_samples"]
        , max_features=PARAMS_BC["max_features"]
        , random_state=0
    )

    trained_model = helper_layer.model_cv_train(x_train, y_train, bagging_model)
    metrics = helper_layer.evaluate_model(x_train, y_train, x_test, y_test, trained_model)
    return PARAMS_BC, metrics, trained_model


# K Means Classifier => Unused?
def kmeans(X_train, y_train):
    kmeans = KMeans(n_clusters=9, random_state=0).fit(X)
    kmeans.labels_
    kmeans.predict([[0, 0], [12, 3]])


def gmlvq(train_data, train_labels):
    # temporarily hardcoded values
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
    return (model)
