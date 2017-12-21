import pickle

import numpy as np
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from ensemble import AdaBoostClassifier

if __name__ == "__main__":

    ep = 0.000001

    def get_data(dataset_X, dataset_y):
        pkl_load_X = open(dataset_X, "rb")
        pkl_load_y = open(dataset_y, "rb")

        X = pickle.load(pkl_load_X)
        y = pickle.load(pkl_load_y)

        print("X = ", X.shape)
        print("y = ", y.shape)
        return X, y


    def train(train_X, train_y):
        weak_classifier = DecisionTreeClassifier(max_depth=3)
        ada = AdaBoostClassifier(weak_classifier, 5)
        ada.fit(train_X, train_y)
        result = ada.predict(train_X)
        diff = np.abs(result - train_y)
        diff[diff > ep] = 1
        t = np.sum(diff)
        print("错误预测的个数为： ", t)
        target_names = ['人脸', '非人脸']
        report = (classification_report(train_y, result, target_names=target_names))

        re_path = "/home/sun/ComputerScience/MachineLearning/Experiments/Experiment_three/ML2017-lab-03/report.txt"
        write_report(re_path, report)
        return ada

    def test(adaboost, valid_x, valid_y):
        prediction = adaboost.predict(valid_x)
        diff = np.abs(prediction - valid_y)
        diff[diff > ep] = 1
        t = np.sum(diff)
        print("错误预测的个数为： ", t)
        target_names = ['人脸 ', '非人脸']
        report = (classification_report(valid_y, prediction, target_names=target_names))
        
        re_path = "/home/sun/ComputerScience/MachineLearning/Experiments/Experiment_three/ML2017-lab-03/report.txt"
        write_report(re_path, report)
        

    def write_report(file, report):
        with open(file, "a") as f:
            f.write(report)

    dataset_X_file = "/home/sun/ComputerScience/MachineLearning/Experiments/Experiment_three/ML2017-lab-03/datasets/features/X_train"
    dataset_y_file = "/home/sun/ComputerScience/MachineLearning/Experiments/Experiment_three/ML2017-lab-03/datasets/features/y_train"
    train_X, train_y = get_data(dataset_X_file, dataset_y_file)
    print("train_y = ", train_y.shape)

    print("**************1111************")
    print("           加载数据完成          ")
    print("*******************************")

    Adaboost = train(train_X, train_y)

    vaild_x_file = "/home/sun/ComputerScience/MachineLearning/Experiments/Experiment_three/ML2017-lab-03/datasets/valid/x_valid"
    valid_y_file = "/home/sun/ComputerScience/MachineLearning/Experiments/Experiment_three/ML2017-lab-03/datasets/valid/y_valid"
    valid_x, valid_y = get_data(vaild_x_file, valid_y_file)
    
    test(Adaboost, valid_x, valid_y)
