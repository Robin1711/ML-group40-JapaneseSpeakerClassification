import time
from itertools import chain
import mlflow
from sklearn.metrics import classification_report

# Self defined modules
import data_formatting
import preprocessing
import model_layer

LABEL_TYPE = "timeseries"       # [salad, timeseries]
PREPROCESSING_STEPS = []
MODEL_TYPE = "Bagging"         # [RandomForest, SVC, Ensemble, Bagging]
SIGNAL_LENGTH = 22


def run_gmlvq(test_data, test_labels, model):
    predicted_labels = model.predict(test_data)
    print("GMLVQ:\n" + classification_report(test_labels, predicted_labels))
    return(predicted_labels)


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
    D[1] = data_formatting.transform_labelvectors_to_labels(D[1])
    D[3] = data_formatting.transform_labelvectors_to_labels(D[3])

    if LABEL_TYPE == "timeseries":
        D[0] = data_formatting.pad_truncate_transpose_data(D[0], signal_length=SIGNAL_LENGTH)
        D[0] = [list(chain(*example)) for example in D[0]]
        D[2] = data_formatting.pad_truncate_transpose_data(D[2], signal_length=SIGNAL_LENGTH)
        D[2] = [list(chain(*example)) for example in D[2]]

    print("Loaded data")
    params, metrics = dict(), dict()

    if MODEL_TYPE == "RandomForest":
        params, metrics, trained_model = model_layer.random_forest(D)
    elif MODEL_TYPE == "SVC":
        params, metrics, trained_model = model_layer.support_vector_machine(D)
    elif MODEL_TYPE == "Ensemble":
        params, metrics, trained_model = model_layer.ensemble_classifier(D)
    elif MODEL_TYPE == "Bagging":
        params, metrics, trained_model = model_layer.bagging_classifier(D)
    else:
        exit("Pleas specify a model")

    params["label_type"] = LABEL_TYPE
    params["preproccessing_steps"] = PREPROCESSING_STEPS
    log_results(params, metrics)

    end_time = time.time()
    print(f'\nElapsed time: {end_time - start_time}')
