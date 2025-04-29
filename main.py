from SVM import SVM

def test_svm():
    svm = SVM(kernel='rbf', gamma=1, C=1)
    X = [[1, 2], [2, 3], [3, 3]]
    y = [1, 1, -1]
    svm.fit(X,y)

if __name__ == '__main__':
    test_svm()
