from text_classification import generate_model, model_validation
from modify_dataset import modify_dataset_and_raw_data_with_percentage_size_to_keep
from modify_dataset import modify_dataset_select_features

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression

from joblib import load
from text_preprocessing import _load_data
from codecarbon import EmissionsTracker
import csv
import time
import random

from click import progressbar
import click

RESULTS_FILE = 'results.csv'
RESULTS_HEADER = [
    'algorithm',
    'RQ',
    'experiment_id',
    'iteration',
    'no_datapoints',
    'no_features',
    'preprocessing_energy(J)',
    'preprocessing_time(s)',
    'train_energy(J)',
    'train_time(s)',
    'predict_energy(J)',
    'predict_time(s)',
    'datatype',
    'accuracy',
    'precision',
    'recall',
    'f1',
]

results = []

raw_data = _load_data()
preprocessed_data = load('output/preprocessed_data.joblib')


NUMBER_OF_EXPERIMENTAL_RUNS = 30
SLEEP_TIME = 5
CLASSIFIERS = {
    'SVM': SVC(class_weight="balanced"),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': ComplementNB(),
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(class_weight="balanced"),
    'AdaBoost': AdaBoostClassifier(),
    'Bagging Classifier': BaggingClassifier()
}

dataset_size_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
featureset_size_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def energy_stats(energy_consumption_kwh, energy_tracker):
    """Extract and compute energy metrics from codecarbon Energy Tracker.
        IMPORTANT: this function should be called right after stopping the tracker.
    """
    energy_consumption_joules = energy_consumption_kwh * 1000 * 3600 #Joules
    duration = energy_tracker._last_measured_time - energy_tracker._start_time
    return energy_consumption_joules, duration

def write_header(filename):
    with open(filename, mode='w') as results_file:
        result_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(RESULTS_HEADER)

def write_result(result, filename):
    with open(filename, mode='a') as results_file:
        result_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        result_writer.writerow(result)


def run_experiment(RQ, iteration, experiment_id, classifier_name, dataset_percentage, featureset_percentage):
    classifier = CLASSIFIERS[classifier_name]
    
    print(f"Starting Experiment: {experiment_id},"
          f"\n research question {RQ},"
          f"\n iteration {iteration},"
          f"\n classifier {classifier},"
          f"\n dataset_percentage {dataset_percentage},"
          f"\n featureset_percentage {featureset_percentage}"
    )
    time.sleep(SLEEP_TIME)

    preprocessing_tracker = EmissionsTracker(save_to_file=False)
    #### START TIMED PREPROCESSING SECTION ####
    preprocessing_tracker.start()
    # Danger: dataset_percentage
    modified_preprocessed_data, modified_raw_data = modify_dataset_and_raw_data_with_percentage_size_to_keep(
        preprocessed_data, raw_data, dataset_percentage)

    # feature selection
    modified_preprocessed_data, modified_raw_data = modify_dataset_select_features(
        modified_preprocessed_data, modified_raw_data, featureset_percentage
    )
    preprocessing_energy_consumption_kwh = preprocessing_tracker.stop()
    #### STOP TIMED PREPROCESSING SECTION ####
    preprocessing_energy_consumption, preprocessing_duration = energy_stats(preprocessing_energy_consumption_kwh,
                                                                            preprocessing_tracker)

    training_tracker = EmissionsTracker(save_to_file=False)
    #### START TIMED TRAINING SECTION ####
    training_tracker.start()
    classifier, X_train, X_test, y_train, y_test, test_messages = generate_model(classifier, modified_raw_data,
                                                                                 modified_preprocessed_data)
    training_energy_consumption_kwh = training_tracker.stop()
    #### STOP TIMED TRAINING SECTION ####
    training_energy_consumption, training_duration = energy_stats(training_energy_consumption_kwh, training_tracker)

    predict_tracker = EmissionsTracker(save_to_file=False)
    #### START TIMED PREDICTION SECTION ####
    predict_tracker.start()
    _, scores, report = model_validation(classifier, X_test, y_test)
    print (scores)
    print (report)
    predict_energy_consumption_kwh = predict_tracker.stop()
    #### STOP TIMED PREDICTION SECTION ####
    predict_energy_consumption, predict_duration = energy_stats(predict_energy_consumption_kwh, predict_tracker)

    number_of_datapoints = len(y_train)
    number_of_features = X_train.shape[1]

    print(f"Experiment ID {experiment_id}")
    print(f"Run {iteration}")
    print(f"  Energy Consumption: {training_energy_consumption} Joules")
    print(f"  Duration: {training_duration} seconds")

    result_row = [
        classifier_name,
        RQ,
        experiment_id,
        iteration,
        number_of_datapoints,
        number_of_features,
        preprocessing_energy_consumption,
        preprocessing_duration,
        training_energy_consumption,
        training_duration,
        predict_energy_consumption,
        predict_duration,
        "float64", #datatype
        scores['accuracy'],
        scores['precision'],
        scores['recall'],
        scores['f1'],
    ]
    results.append(result_row)
    write_result(result_row, RESULTS_FILE)

def collect_previous_experiments(filename):
    try:
        with open(filename, mode='r') as results_file:
            result_reader = csv.DictReader(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            return list(result_reader)
    except FileNotFoundError:
        return None

def _compute_experiment_hash(exp):
    return f"{exp['experiment_id']}_{exp['iteration']}"

def run_experiment_batch(experiments):
    previous_experiments = collect_previous_experiments(RESULTS_FILE)
    if previous_experiments:
        previous_ids = [_compute_experiment_hash(exp) for exp in previous_experiments]
        print(f"There were {len(previous_ids)} experiments in {RESULTS_FILE}."
              f" Skipping ids {previous_ids}.")
        experiments = [exp for exp in experiments if _compute_experiment_hash(exp) not in previous_ids]
    else:
        write_header('results.csv')
    print(f"Remaining experiments: {len(experiments)}.")
    random.shuffle(experiments)
    with progressbar(experiments) as bar:
        for experiment in bar:
            print("\n")
            run_experiment(**experiment)

def fibonacci(n):
    if n<= 0:
        print("Incorrect input")
    # First Fibonacci number is 0
    elif n == 1:
        return 0
    # Second Fibonacci number is 1
    elif n == 2:
        return 1
    else:
        return fibonacci(n-1)+fibonacci(n-2)

# default values
featureset_percentage = 100
dataset_percentage = 100
# initial values
experiment_id=0
experiments = []

# run classification experiment
for classifier_name in CLASSIFIERS.keys():
    RQ="2.1"
    for dataset_percentage in dataset_size_percentages:
        experiment_id += 1
        for iteration in range(NUMBER_OF_EXPERIMENTAL_RUNS):
            experiments.append({
                "RQ": RQ,
                "iteration": iteration,
                "experiment_id": experiment_id,
                "classifier_name": classifier_name,
                "dataset_percentage": dataset_percentage,
                "featureset_percentage": featureset_percentage,
            })
        
    dataset_percentage = 100

    RQ="2.2"
    for featureset_percentage in featureset_size_percentages:
        experiment_id += 1
        for iteration in range(NUMBER_OF_EXPERIMENTAL_RUNS):
            experiments.append({
                "RQ": RQ,
                "iteration": iteration,
                "experiment_id": experiment_id,
                "classifier_name": classifier_name,
                "dataset_percentage": dataset_percentage,
                "featureset_percentage": featureset_percentage,
            })
    featureset_percentage = 100

fibonacci(35)
run_experiment_batch(experiments)
