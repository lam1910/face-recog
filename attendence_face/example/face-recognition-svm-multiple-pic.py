# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:09:51 2020

@author: lamnn-tt
"""
# Train multiple images per person
# Find and recognize faces in an image using a SVC with scikit-learn

"""
Structure:
        <test_image>.jpg
        <train_dir>/
            <person_1>/
                <person_1_face-1>.jpg
                <person_1_face-2>.jpg
                .
                .
                <person_1_face-n>.jpg
           <person_2>/
                <person_2_face-1>.jpg
                <person_2_face-2>.jpg
                .
                .
                <person_2_face-n>.jpg
            .
            .
            <person_n>/
                <person_n_face-1>.jpg
                <person_n_face-2>.jpg
                .
                .
                <person_n_face-n>.jpg
"""
import face_recognition
from sklearn import svm
import os

import pickle

# will return true if successed, false otherwise file path and name must be 
# new, otherwise return false
def export_bin(obj_to_export, file_path = 'models/finalized_model.sav'):
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
    
def load_bin(file_path):
    return pickle.load(open(file_path, 'rb'))

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
train_dir = os.listdir('dataset/train_fol/')

# Loop through each person in the training directory
for person in train_dir:
    pix = os.listdir("dataset/train_fol/" + person)

    # Loop through each training image for the current person
    for person_img in pix:
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file("dataset/train_fol/" + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image contains exactly one face
        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img + " was skipped and can't be used for training")

# Create and train the SVC classifier
clf = svm.SVC(gamma='scale')
clf.fit(encodings,names)

# TODO: export to a binary for faster recognition time. DONE
export_bin(clf, 'models/test-model-famous-ppl.sav')
clf = load_bin('models/test-model-famous-ppl.sav')

# cannot detect unknown yet, have to double check to change that
# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file('dataset/unknown_faces/duong-hong-son.jpg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
no = len(face_locations)
print("Number of faces detected: ", no)

# Predict all the faces in the test image using the trained classifier
print("Found:")
for i in range(no):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_enc])
    # TODO: double check here, if distance is too much change back to unknown. DONE
    path_to_abs_true = os.path.join('dataset/known_faces', name[0] + '.jpg')
    per_true_img = face_recognition.load_image_file(path_to_abs_true)
    true_face = face_recognition.face_encodings(per_true_img)
    true_result = face_recognition.face_distance(true_face, test_image_enc)
    if true_result[0] < 0.45:
        print(*name)
    else:
        print('Unkonwn')