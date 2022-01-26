"""
Train the model using different algorithms.
Gets 1 variable as input: algorithm name. See 'classifiers' dictionary.
Creates 3 files in output: `accuracy_scores.png`,
`model.joblib`, and `misclassified_msgs.txt`.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt
from text_preprocessing import _load_data
from sys import argv

pd.set_option('display.max_colwidth', None)


def my_train_test_split(*datasets):
    '''
    Split dataset into training and test sets. We use a 70/30 split.
    The last dataset is used to stratify, to avoid test_sets without spam labels.
    '''
    return train_test_split(*datasets, test_size=0.3, stratify=datasets[-1], random_state=101)

def train_classifier(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)

def predict_labels(classifier, X_test):
    return classifier.predict(X_test)


def generate_model(classifier, raw_data, preprocessed_data):
    (X_train, X_test,
     _, test_messages,
     y_train, y_test) = my_train_test_split(preprocessed_data,
                                            raw_data['message'],
                                            raw_data['label'])
                                             

    # classifier = classifiers.get(algorithm)
    train_classifier(classifier, X_train, y_train)
    return classifier, X_train, X_test, y_train, y_test, test_messages

def model_validation(classifier, X_test, y_test):
    predictions = predict_labels(classifier, X_test)
    report = classification_report(y_test, predictions)
    scores = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label='spam'),
        "recall": recall_score(y_test, predictions, pos_label='spam'),
        "f1": f1_score(y_test, predictions, pos_label='spam')
    }
    return predictions, scores, report
    
def main():
    algorithm = str(argv[1])
    raw_data = _load_data()
    preprocessed_data = load('output/preprocessed_data.joblib')
    

    pred_scores = dict()
    pred = dict()
    # save misclassified messages
    file = open('output/misclassified_msgs.txt', 'a', encoding='utf-8')
    key = algorithm
    classifier, X_train, X_test, y_train, y_test, test_messages =  generate_model(algorithm, raw_data, preprocessed_data)
    
    pred[key], scores, report = model_validation(classifier, X_test, y_test)
    pred_scores[key] = [scores["accuracy"]]
    print('\n############### ' + key + ' ###############\n')
    print(report)

    # write misclassified messages into a new text file
    file.write('\n#################### ' + key + ' ####################\n')
    file.write('\nMisclassified Spam:\n\n')
    for msg in test_messages[y_test < pred[key]]:
        file.write(msg)
        file.write('\n')
    file.write('\nMisclassified Ham:\n\n')
    for msg in test_messages[y_test > pred[key]]:
        file.write(msg)
        file.write('\n')
    file.close()

    print('\n############### Accuracy Scores ###############')
    accuracy = pd.DataFrame.from_dict(pred_scores, orient='index', columns=['Accuracy Rate'])
    print('\n')
    print(accuracy)
    print('\n')

    #plot accuracy scores in a bar plot
    accuracy.plot(kind='bar', ylim=(0.85, 1.0), edgecolor='black', figsize=(10, 5))
    plt.ylabel('Accuracy Score')
    plt.title('Distribution by Classifier')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("output/accuracy_scores.png")

    # Store "best" classifier
    dump(classifier, 'output/'+algorithm+'_model.joblib')

if __name__ == "__main__":
    main()