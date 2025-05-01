from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from SVM import SVM
from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
import time

def load_split_data():
    wine_quality = fetch_ucirepo(id=186)

    X = wine_quality.data.features.to_numpy()
    y = wine_quality.data.targets.to_numpy().ravel()

    samples_count = 4000
    np.random.seed(42)
    indices = np.random.permutation(X.shape[0])[:samples_count]
    X = X[indices]
    y = y[indices]

    y[y <= 5] = -1
    y[y > 5] = 1

    test_fraction = 0.2
    test_number = int(test_fraction * samples_count)

    X_test = X[:test_number]
    y_test = y[:test_number]
    X_train = X[test_number:]
    y_train = y[test_number:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 1
    else:
        recall = TP / (TP + FN)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1_score


def test_svm_hiperparams_rbf():
    X_train, y_train, X_test, y_test = load_split_data()

    # C_s = [0.1,1,10,100,1000,10000]
    # gamma_s = [0.0001,0.001,0.01,0.1,1]
    # C_s = [0.001,0.01, 0.1, 1]
    # gamma_s = [1,10,100,1000]
    # C_s = [100]
    # gamma_s = [0.1]
    C_s = [100]
    gamma_s = [0.1]
    recall_s = np.zeros((len(C_s), len(gamma_s)))
    precision_s = np.zeros((len(C_s), len(gamma_s)))
    f1_score_s = np.zeros((len(C_s), len(gamma_s)))
    accuracy_s = np.zeros((len(C_s), len(gamma_s)))

    for i in range(len(C_s)):
        for j in range(len(gamma_s)):
            svm = SVM(kernel='rbf', gamma=gamma_s[j], C=C_s[i], coef0=1, degree=3)
            start = time.time()
            svm.fit(X_train, y_train)
            end = time.time()
            print(f"time: {end - start:.6f} s")
            y_pred = svm.predict(X_test)
            accuracy, precision, recall, f1_score = calculate_metrics(y_pred, y_test)
            print(gamma_s[j], C_s[i])
            print(accuracy, precision, recall, f1_score)
            accuracy_s[i,j] = accuracy
            precision_s[i,j] = precision
            recall_s[i,j] = recall
            f1_score_s[i,j] = f1_score

    plt.figure(figsize=(8, 6))
    sns.heatmap(accuracy_s, annot=True, fmt=".4f", xticklabels=gamma_s, yticklabels=C_s, cmap='viridis')
    plt.xlabel("Gamma")
    plt.ylabel("C")
    plt.title("Accuracy w zależności od C i gamma")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(precision_s, annot=True, fmt=".4f", xticklabels=gamma_s, yticklabels=C_s, cmap='viridis')
    plt.xlabel("Gamma")
    plt.ylabel("C")
    plt.title("Precision w zależności od C i gamma")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(recall_s, annot=True, fmt=".4f", xticklabels=gamma_s, yticklabels=C_s, cmap='viridis')
    plt.xlabel("Gamma")
    plt.ylabel("C")
    plt.title("Recall w zależności od C i gamma")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(f1_score_s, annot=True, fmt=".4f", xticklabels=gamma_s, yticklabels=C_s, cmap='viridis')
    plt.xlabel("Gamma")
    plt.ylabel("C")
    plt.title("F1-score w zależności od C i gamma")
    plt.show()


def test_log_regr():
    X_train, y_train, X_test, y_test = load_split_data()
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    model = LogisticRegression()
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print(f"time: {end - start:.6f} s")
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1_score = calculate_metrics(y_pred, y_test)
    print(accuracy, precision, recall, f1_score)

def test_knn():
    X_train, y_train, X_test, y_test = load_split_data()
    model = KNeighborsClassifier()
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print(f"time: {end - start:.6f} s")
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1_score = calculate_metrics(y_pred, y_test)
    print(accuracy, precision, recall, f1_score)

def test_dummy():
    X_train, y_train, X_test, y_test = load_split_data()
    model = DummyClassifier(strategy='uniform', random_state=0)
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    print(f"time: {end - start:.6f} s")
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1_score = calculate_metrics(y_pred, y_test)
    print(accuracy, precision, recall, f1_score)

if __name__ == '__main__':
    test_svm_hiperparams_rbf()
    # test_log_regr()
    # test_knn()
    # test_dummy()
