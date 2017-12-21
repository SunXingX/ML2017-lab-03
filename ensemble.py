import pickle
import os

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        self.G={}
        self.alpha={}

        save_G_path = "/home/sun/ComputerScience/MachineLearning/Experiments/Experiment_three/ML2017-lab-03/self_G"
        save_alpha_path = "/home/sun/ComputerScience/MachineLearning/Experiments/Experiment_three/ML2017-lab-03/self_alpha"

        if os.path.exists(save_G_path):
            pkl_G = open(save_G_path, "rb")
            self.G = pickle.load(pkl_G)
        if os.path.exists(save_alpha_path):
            pkl_alpha = open(save_alpha_path, "rb")
            self.alpha = pickle.load(pkl_alpha)

        if self.G and self.alpha:
            return True

        n_samples = X.shape[0]
        n_features = X.shape[1]
        #initilize the W
        self.W = np.ones(n_samples) / n_samples
        
        for i in range(self.n_weakers_limit):
            self.G.setdefault(i)
            self.alpha.setdefault(i)

        #training a basic classifier
        for i in range(self.n_weakers_limit):
            self.G[i] = self.weak_classifier
            self.G[i].fit(X, y, sample_weight = self.W)
            train_score = self.G[i].score(X, y)
            e = 1 - train_score
            print("e = ", e)

            self.alpha[i] = 1 / 2 * np.log((1 - e) / e)
           
            prediction = self.G[i].predict(X)
            prediction = prediction.reshape((prediction.shape[0], 1))
            print("prediction = ", prediction.shape)
        
            self.W = self.W.reshape(self.W.shape[0], 1)
            print("self.W = ", self.W.shape)
            tmp = np.exp(-self.alpha[i] * np.multiply(y, prediction))
            print("tmp = ", tmp.shape)
            Zi = np.sum(np.dot(tmp.T, self.W))
            
            self.W = np.multiply(self.W, tmp) / Zi

            self.W = self.W.reshape((self.W.shape[0],))
           
           
            print("alpha", self.alpha[i])
            print("self.W  reshap = ",self.W.shape)
            print("Zi = ",Zi)
            print("  ")
        
        if not os.path.exists(save_G_path):
            pkl_G = open(save_G_path, "wb")
            pickle.dump(self.G, pkl_G)
        if not os.path.exists(save_alpha_path):
            pkl_alpha = open(save_alpha_path, "wb")
            pickle.dump(self.alpha, pkl_alpha)
        
        pass


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        fx = 0

        for i in range(self.n_weakers_limit):
            fx = fx + self.alpha[i] * self.G[i].predict(X)
            
        fx = fx.reshape((fx.shape[0], 1))
        return fx

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        final_prediction = self.sign(self.predict_scores(X))
        print("final_prediction = ", final_prediction.shape)
        return final_prediction

    def sign(self, x):
        sig_value = 1 / (1 + np.exp(-x))
        print(sig_value)
        sig_value[sig_value > 0.5] = 1
        sig_value[sig_value <= 0.5] = -1
        return sig_value

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
