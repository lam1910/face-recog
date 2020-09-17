# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 11:30:56 2020

@author: Lam Nguyen Ngoc
"""
import face_recognition
import os
import pandas as pd
from sqlalchemy.exc import ArgumentError


class LoadTrainset:
    trainset_path = 'dataset/train_fol/'
    encodings = []
    names = []

    def settrainset_path(self, value):
        self.trainset_path = value
    
    # load the whole image sets for the first time. As the size increased, 
    # time spent to load the image set this way will become extremely long. 
    # because of that, there will be some functions to support read and write 
    # to another source. At 10/9/2020, support read and write from database 
    # and from an excel file. (Update afternoon 10/9/2020, have not added the 
    # packages for db connection to requirements yet)                
    def load(self):
        train_dir = os.listdir(self.trainset_path)
        encodings = []
        names = []
        # Loop through each person in the training directory
        for person in train_dir:
            pix = os.listdir(self.trainset_path + person)
            # Loop through each training image for the current person
            for person_img in pix:
                # Get the face encodings for the face in each image file
                face = face_recognition.load_image_file(self.trainset_path + person + '/' + person_img)
                face_bounding_boxes = face_recognition.face_locations(face)
        
                #If training image contains exactly one face
                if len(face_bounding_boxes) == 1:
                    face_enc = face_recognition.face_encodings(face)[0]
                    # Add face encoding for current image with corresponding label (name) to the training data
                    encodings.append(face_enc)
                    names.append(person)
                else:
                    print(person + "/" + person_img + " was skipped and can't be used for training")
        self.encodings = encodings
        self.names = names
    
    # option 1: save to a table of a database of your choice, provided that 
    # the connection param is correct and the database is up and running
    def save_to_db(self, name, connection):
        df = pd.DataFrame(self.encodings)
        df['Name'] = self.names
        try:
            df.to_sql(name, connection, index=False)
        except AttributeError:
            print('Second arg had to be a Connection class to database')
            return False
        except ArgumentError:
            print('Could not find the server or unkown error. If you parse a string here, it is likely because of syntax error in string')
            return False
        except pd.ConnectionError:
            print('Could not find the server or unkown error.')
            return False
        return True
    
    # option 2: save to an excel file, provided you added the correct 
    # file path into the name param
    def save_to_file(self, name, connection=None):
        df = pd.DataFrame(self.encodings)
        df['Name'] = self.names
        try:
            df.to_excel(name, index=False)
        except Exception:
            print('Unexpected errors. Return False')
            return False
        return True
    
    # method to load dataframe from the database, again with functioning 
    # connection
    def load_from_db(self, name, connection):
         df = pd.read_sql(name, connection)
         self.encodings = df.iloc[:, :128].values.tolist()
         self.names = df['Name'].values.tolist()
    
    # method to load dataframe from the database, again with correct name
    def load_from_file(self, name):
        df = pd.read_excel(name)
        self.encodings = df.iloc[:, :128].values.tolist()
        self.names = df['Name'].values.tolist()
