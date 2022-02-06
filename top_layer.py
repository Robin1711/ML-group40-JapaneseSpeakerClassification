import time
import mlflow
from sklearn.metrics import classification_report
import numpy as np

# Self defined modules
import data_formatting
import preprocessing
import model_layer

LABEL_TYPE = "timeseries"       # [salad, timeseries]
SIGNAL_LENGTH = 22
PREPROCESSING_STEPS = [["pad","BACK",15],["truncate","BACK",15],["transpose"],["flatten"],["onehot-to-labels"]],  #["pad","truncate","transpose","flatten","auto_reg_coeff","pca","onehot-to-labels"]
MODEL_TYPE = "Bagging"         # [RandomForest, SVC, Ensemble, Bagging, NaiveBayes, ESN]



def log_results(params, metrics):
    print(f"\nLOG:\n{params['classifier']}\n"
          f"\tTrain Accuracy: {metrics['accuracy_train']}\n"
          f"\tTest Accuracy: {metrics['accuracy_test']}\n"
          f"\tTest F-Score: {metrics['fScore_test']}\n"
          f"\tTest Precision: {metrics['precision_test']}")

    print(f"\nCONFUSION MATRIX:\n{metrics.pop('confusion_matrix')}")

    mlflow.log_params(params)           # Log parameters
    mlflow.log_metrics(metrics)         # Log metrics
    mlflow.log_artifact("./models.py")  # Save a copy of models


if __name__ == '__main__':
    start_time = time.time()
    print("Starting..")

    D = preprocessing.get_data(label_type=LABEL_TYPE, preproccessing_steps=PREPROCESSING_STEPS)

    if MODEL_TYPE == "RandomForest":
        params, metrics, trained_model,_ = model_layer.random_forest(D)
    elif MODEL_TYPE == "SVC":
        params, metrics, trained_model,_ = model_layer.support_vector_machine(D)
    elif MODEL_TYPE == "Ensemble":
        params, metrics, trained_model,_ = model_layer.ensemble_classifier(D)
    elif MODEL_TYPE == "Bagging":
        params, metrics, trained_model,_ = model_layer.bagging_classifier(D)
    elif MODEL_TYPE == "NaiveBayes":
        params, metrics, trained_model,_ = model_layer.naive_bayes(D)
    elif MODEL_TYPE == "ESN":
        params, metrics, trained_model,_ = model_layer.echo_state_network(np.asarray(D))
    elif MODEL_TYPE == "GMLVQ":
        params, metrics, trained_model,_ = model_layer.gmlvq(D)
    else:
        exit("Please specify a model")

    params["label_type"] = LABEL_TYPE
    params["preproccessing_steps"] = PREPROCESSING_STEPS
    log_results(params, metrics)

    end_time = time.time()
    print(f'\nElapsed time: {end_time - start_time}')



#feature_stationarity = np.zeros(12)
#
    #for i, recording in enumerate(D[0]):
    #    for j, feature in enumerate(recording):
    #        stationarityTest = adfuller(D[0][i][j], autolag='AIC')
    #        if stationarityTest[1] <= 0.05:
    #            feature_stationarity[j] += 1
    #
    ## Check the value of p-values
    #print(feature_stationarity)