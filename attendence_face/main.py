# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:29:01 2020

@author: lam Nguyen Ngoc
"""
# this only for development and testing phrase, will create a new main 
# for live version
from core.load_trainset import LoadTrainset
from core.check_in import CheckIn
from core.train_model import Trainer

# read df
test_dataset = LoadTrainset()
# test_dataset.load()
test_dataset.load_from_file('dataset/test-2.xlsx')

# model
clf = Trainer(test_dataset.encodings, test_dataset.names, 'rf')
new_clf = clf.train_model(wr_out=False, file_name=None, svc_kernel='sigmoid',
                          svc_decision_function_shape='ovo', svc_gamma='auto',
                          n_estimators=100, rf_crit='gini',
                          max_feat='auto', warm_start=False)

# test write model out
# clf.train_model(wr_out=True, file_name='test-model-rf.sav', svc_kernel='rbf', 
#                 svc_decision_function_shape='ovo', svc_gamma='auto', 
#                 n_estimators=100, rf_crit='gini', 
#                 max_feat='auto', warm_start=False)
# test_clf = clf.load_model_in('models/test-model-rf.sav')

checks = CheckIn()

checks.__setknown_face_encodings__(test_dataset.encodings)
checks.__setknown_face_names__(test_dataset.names)

# actual run
checks.face_recognize(new_clf)
# checks.face_recognize()
