# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:05:15 2020

@author: lam Nguyen Ngoc
"""
import os
import pickle


# will return true if succeed, false otherwise file path and name must be
# new, otherwise return false
def export_bin(obj_to_export, file_path = 'finalized_model.sav'):
    file_path = os.path.join('models', file_path)
    try:
        pickle.dump(obj_to_export, open(file_path, 'w+b'))
        print('Successfully write out to: ' + file_path + '.')
        return True
    except FileNotFoundError:
        pc = os.path.split(file_path)
        os.mkdir(pc[0])
        pickle.dump(obj_to_export, open(file_path, 'w+b'))
        print('Successfully write out to: ' + file_path + '.')
        return True
    except Exception:
        print('Writing out to: ' + file_path + ' encounter some errors. Check syntax.')
        return False


# load something from file path (will use to load an outside file contains 
# model)
def load_bin(file_path):
    return pickle.load(open(file_path, 'rb'))
