# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 15:08:12 2020

@author: lam Nguyen Ngoc
"""
# need to check back ưhen using svm to classify ppl as svm will return the 
# best fit even when that person isn't on the database.
# import the face distance only to make the program more efficient
from face_recognition import face_distance
import numpy as np


# check the result from the model to see whether it fall in the case of 
# not on the database but still return as one in the database and similar face
# in the database that confused the system.
#   name_predicted: label that the model give to that face in test img
#   true_face: the encoding of 1 face of the person named name_predicted in 
#       the database. Due to the contruction of the db, how classifier model 
#       works and the fact that we will use this function only after getting 
#       the name from the classifier, it will not be put in any try-catch stmt
#   test_image_enc: the encoding of the face that we get from the camera
#   threshold: maximum distance between true_face and test_image_enc that we 
#       determine to conclude that those two pictures are from the same person
def double_check_result(name_predicted, true_face, test_image_enc, threshold = 0.45):
    # type of true_face param will either be a list, which the C order (default)
    # of the asarray method will convert to a ndarray vector, or a ndarray 
    # vector from the beginning. Because of that, there is no need to check
    # whether the new_true_face is in the right dimension or not
    
    # convert input of true_face into ndarray
    if type(true_face) != np.ndarray:
        new_true_face = np.asarray(true_face)
    else:
        new_true_face = true_face
    
    # if the name fed in cannot be understand by the system, return 'Unknown'
    # having numpy.str_ type because of how the return of predict method ò sklearn
    # having str since it is a built-in type of string representation of 
    # python and in case someone try to convert back the np.str_ to str
    if name_predicted == 'Unknown' or type(name_predicted) not in (np.str_, str):
        return 'Unknown'
    
    # else
    # calculate the distance between the true face and the face encoding 
    # from the test source
    true_result = face_distance([new_true_face], test_image_enc)

    # if it is below the threshold, meaning that we accept that the test img 
    # contains the person with the name already predicted
    if true_result[0] < threshold:
        return name_predicted
    # otherwise, the person in the test img is not consider a known person 
    # in the db, also return unknown
    else:
        return 'Unknown'
