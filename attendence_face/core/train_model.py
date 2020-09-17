# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:55:21 2020

@author: lam Nguyen Ngoc
"""

# attempt to having multiple picture for 1 person
# Train multiple images per person
# Adding ANN to method
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from utilis.model_io import load_bin, export_bin
import warnings
import keras
from keras.models import Sequential
from keras.layers import Dense
import random


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

    # define the artificial neural network
    def init_ann(self, input_dim, unit_hidden_layer, unit_output, no_of_hidden_layer=1, kernel_initializer='uniform',
                 activation='relu', output_activation='softmax', optimizer='adam', loss='binary_crossentropy',
                 metrics=['accuracy']):
        clf = Sequential()
        # put the input layer and first layer
        clf.add(Dense(units=unit_hidden_layer, input_dim=input_dim, kernel_initializer=kernel_initializer,
                      activation=activation))
        # add hidden layer
        for i in range(no_of_hidden_layer):
            clf.add(Dense(units=unit_hidden_layer, kernel_initializer=kernel_initializer, activation=activation))
        # output layer
        clf.add(Dense(units=unit_output, kernel_initializer=kernel_initializer, activation=output_activation))
        # compiling ANN
        clf.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return clf

    # train model + decide on how to store the model
    def train_model(self, wr_out=False, file_name=None,
                    svc_kernel='rbf', svc_gamma='scale', svc_decision_function_shape='ovr',
                    n_estimators=100, rf_crit='gini', max_feat='auto',
                    warm_start=False, input_dim=None, unit_hidden_layer=None, unit_output=None, no_of_hidden_layer=1,
                    kernel_initializer='uniform', activation='relu', output_activation='softmax', optimizer='adam',
                    loss='binary_crossentropy', metrics=['accuracy']):
        # dimension for ann
        dim_in = len(self.__X[0])
        dim_out = len(set(self.__label))
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
        # if user choose the deep learning
        elif self.method == 'neural network' or self.method == 'nn' or self.method == 'ann':
            # if input_dim = None
            if input_dim is None:
                input_dim = dim_in
            if unit_output is None:
                unit_output = dim_out
            if unit_hidden_layer is None or unit_hidden_layer < dim_in or unit_hidden_layer > dim_out:
                warnings.warn('The number of units in the hidden layer is not optimal. It should be in between '
                              'dimension of input and dimension of output.', UserWarning, stacklevel=2)
                unit_hidden_layer = random.randrange(dim_out, dim_in)

            try:
                # initialize neural network
                clf = self.init_ann(input_dim, unit_hidden_layer, unit_output, no_of_hidden_layer, kernel_initializer,
                                    activation, output_activation, optimizer, loss, metrics)
                clf.fit(self.__X, self.__label)
            except ValueError:
                print('Arguments for RandomForest is not acceptable for sklearn. Construct with default value')
                clf = self.init_ann(dim_in, unit_hidden_layer, dim_out, 1, 'uniform', 'relu', 'softmax', 'adam',
                                    'binary_crossentropy', ['accuracy'])
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
