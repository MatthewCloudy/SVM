from numpy.ma.core import shape
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from SVM import SVM
from ucimlrepo import fetch_ucirepo
import numpy as np

def load_split_data():
    wine_quality = fetch_ucirepo(id=186)

    X = wine_quality.data.features.to_numpy()[:600]
    y = wine_quality.data.targets.to_numpy().ravel()[:600]
    y[y<5] = -1
    y[y>=5] = 1
    test_fraction = 0.3
    test_number = int(test_fraction * X.shape[0])
    indices = np.random.permutation(X.shape[0])
    test_indices = indices[:test_number]
    train_indices = indices[test_number:]
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    return X_train, y_train, X_test, y_test

def calculate_metrics(y_pred, y_test):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] == 1:
            if y_test[i] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y_test[i] == 1:
                FN += 1
            else:
                TN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1_score

def test_svm():

    X_train, y_train, X_test, y_test = load_split_data()
    svm = SVM(kernel='poly', gamma=0.3, C=10.0, coeff0=1, deg=2)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy, precision, recall, f1_score = calculate_metrics(y_pred, y_test)
    print("My SVM")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1_score)

def test_svm_sklearn():
    X_train, y_train, X_test, y_test = load_split_data()

    svm = SVC(kernel='poly', gamma=0.3, C=10, coef0=1, degree=2)
    svm.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)

    accuracy, precision, recall, f1_score = calculate_metrics(y_pred, y_test)
    print("Scikit SVM")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1_score)

if __name__ == '__main__':
    test_svm()
    test_svm_sklearn()
