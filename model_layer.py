import numpy as np

# Classifiers
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from RC.modules import RC_model

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
PARAMS_NB = {"classifier": "Naive Bayes: Gaussian"}
PARAMS_RC = {"classifier": "Reservior Computing", "n_internal_units": 450, "spectral_radius": 0.59, "leak": 0.6, "connectivity": 0.25, "input_scaling": 0.1, "noise_level": 0.01, "n_drop":5, "mts_rep": "reservoir", "w_ridge_embedding": 10.0, "readout_type":'lin', "w_ridge": 5.0  }

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

def echo_state_network(data):
    x_train, y_train, x_test, y_test = data
    model = RC_model(
        reservoir=None,
        n_internal_units=PARAMS_RC['n_internal_units'],
        spectral_radius=PARAMS_RC['spectral_radius'],
        leak=PARAMS_RC['leak'],
        connectivity=PARAMS_RC['connectivity'],
        input_scaling=PARAMS_RC['input_scaling'],
        noise_level=PARAMS_RC['noise_level'],
        n_drop=PARAMS_RC['n_drop'],
        mts_rep=PARAMS_RC['mts_rep'],
        w_ridge_embedding=PARAMS_RC['w_ridge_embedding'],
        readout_type=PARAMS_RC['readout_type'],
        w_ridge=PARAMS_RC['w_ridge']
    )

    tr_time = model.train(np.asarray(x_train), np.asarray(y_train))
    print('Training time = %.2f seconds'%tr_time)

    accuracy, f1 = model.test(np.asarray(x_test), np.asarray(y_test))
    print('Accuracy = %.3f, F1 = %.3f'%(accuracy, f1))

def naive_bayes(data):
    x_train, y_train, x_test, y_test = data
    model = GaussianNB()
    trained_model = helper_layer.model_cv_train(x_train, y_train, model)
    metrics = helper_layer.evaluate_model(x_train, y_train, x_test, y_test, trained_model)
    return PARAMS_NB, metrics, trained_model

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


def gmlvq(data, max_runs, k, step_size):

    x_train, y_train, x_test, y_test = data

    train_labels = []

    #for entry in y_train:
    #    print(entry)
    #    train_labels.append(entry.index(1.0))

    model = GMLVQ(
        distance_type="adaptive-squared-euclidean",
        activation_type="identity",
        solver_type="waypoint-gradient-descent",
        solver_params={
            "max_runs": max_runs,
            "k": k,
            "step_size": step_size
        },
        random_state=1428
    )

    model.fit(x_train, y_train)

    test_labels = []
    #for entry in y_test:
    #    print(entry)
    #    test_labels.append(entry.index(1.0))


    predicted_labels = model.predict(x_test)
    print("GMLVQ:\n" + classification_report(y_test, predicted_labels))
    return(predicted_labels, model)
