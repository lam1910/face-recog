# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:55:21 2020

@author: lam Nguyen Ngoc
"""

# attempt to having multiple picture for 1 person
# Train multiple images per person
# Adding ANN to method (deprecated) (removed)
# adding adaboost to method
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from utils.model_io import load_bin, export_bin
import warnings
import keras
from keras.models import Sequential
from keras.layers import Dense
import random
import numpy as np


class Trainer:
    __X = None
    __label = None
    method = 'svc'

    def __init__(self, X=None, y=None, method='svc'):
        self.__X = X
        self.__label = y
        self.method = method.lower()

    # including scenario of writeout with name, writeout w/o name, and not writeout
    def write_out_ctrl(self, wr_out=False, model=None, file_name=None):
        if wr_out and file_name != None and type(file_name) == str:
            return export_bin(model, file_name)
        elif wr_out and (file_name == None or type(file_name) != str):
            warnings.warn(
                'While you want to write out the model, you haven\'t specified the path. Default to \'finalized_model.sav\' to the same folder as the main scprit or whatever working folder you are on',
                RuntimeWarning, stacklevel=2)
            return export_bin(model)
        else:
            warnings.warn(
                'You choose not to write the model out. Keep in mind that you will need to fit the model everytime you restart the system.',
                ResourceWarning, stacklevel=1)
            return None

    # train model + decide on how to store the model
    def train_model(self, wr_out=False, file_name=None,
                    svc_kernel='rbf', svc_gamma='scale', svc_decision_function_shape='ovr',
                    n_estimators=100, rf_crit='gini', max_feat='auto',
                    warm_start=False, learning_rate=0.1):
        # if user choose the svm model
        if self.method == 'svc':
            try:
                clf = SVC(kernel=svc_kernel, gamma=svc_gamma, decision_function_shape=svc_decision_function_shape)
                clf.fit(self.__X, self.__label)
            except ValueError:
                print(
                    'Arguments svc_kernel or svc_gamma is not acceptable for sklearn. Default to kernel=\'rbf\' \and \gamma=\'scale\'')
                clf = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr')
                clf.fit(self.__X, self.__label)
            finally:
                final_res = self.write_out_ctrl(wr_out, clf, file_name)
                if final_res is None:
                    print('You have chosen not to write the model out')
                    return clf
        # if user choose the random forest model
        elif self.method == 'random forest' or self.method == 'rf':
            try:
                clf = RandomForestClassifier(n_estimators=n_estimators,
                                             criterion=rf_crit,
                                             max_features=max_feat,
                                             warm_start=warm_start)
                clf.fit(self.__X, self.__label)
            except ValueError:
                print('Arguments for RandomForest is not acceptable for sklearn. Construct with default value')
                clf = RandomForestClassifier(n_estimators=100,
                                             criterion='gini',
                                             max_features='auto',
                                             warm_start=False)
                clf.fit(self.__X, self.__label)
            finally:
                final_res = self.write_out_ctrl(wr_out, clf, file_name)
                if final_res is None:
                    print('You have chosen not to write the model out. Return the model now')
                    return clf
        # if method chose was adaboost
        elif self.method == 'ada'or self.method == 'adab':
            try:
                clf = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
                clf.fit(self.__X, self.__label)
            except ValueError:
                print('Arguments for AdaBoost is not acceptable for sklearn. Construct with default value')
                clf = AdaBoostClassifier(n_estimators=50, learning_rate=0.1)
                clf.fit(self.__X, self.__label)
            finally:
                final_res = self.write_out_ctrl(wr_out, clf, file_name)
                if final_res is None:
                    print('You have chosen not to write the model out. Return the model now')
                    return clf
        # default using svm model
        else:
            warnings.warn(
                'You have not choosed any model or the method attribute is unkonwn. Program will change to the default mode which is the default parameter of SVC with no writing out.',
                RuntimeWarning, stacklevel=2)
            print(
                'Arguments svc_kernel or svc_gamma is not acceptable for sklearn. Default to kernel=\'rbf\' \and \gamma=\'scale\'')
            clf = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr')
            clf.fit(self.__X, self.__label)
            self.write_out_ctrl(False)
            return clf

    # support function get a model from file_path
    def load_model_in(self, file_path):
        warnings.warn('You are about to load a model from an outside source into a variable.',
                      UserWarning, stacklevel=2)
        return load_bin(file_path)
