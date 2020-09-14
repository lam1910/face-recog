# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:55:21 2020

@author: lam Nguyen Ngoc
@source: https://github.com/ageitgey/face_recognition/blob/master/examples/
"""
import face_recognition
import cv2
import numpy as np
from utilis.check_back_face import double_check_result
from utilis.convert_to_no_accent import convert


class CheckIn():
    known_face_encodings = []
    known_face_names = []
    # Get a reference to webcam #0 (the default one)
    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    
    def __setknown_face_encodings__(self, value):
       self.known_face_encodings = value
       
    def __setknown_face_names__(self, value):
       self.known_face_names = value
       
    def get_face_name(self, face_enc):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(self.known_face_encodings, face_enc, tolerance=0.45)
        name = "Unknown"
    
        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_enc)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = self.known_face_names[best_match_index]
        
        return name
    
    def get_face_name_by_model(self, face_enc, clf):
        try:
            name = clf.predict([face_enc])[0]
            print("Name inputed: {} of type {}".format(name, type(name)))
            index_of_name = self.known_face_names.index(name)
            name_after_check = double_check_result(name, self.known_face_encodings[index_of_name], face_enc)
        # every errors that can be caught now included Errors that does not 
        # related to data structure, algo or type of param. Futhermore, those 
        # errors should be pretty rare so throw out the unknown tag for now.
        except Exception:
            name_after_check = 'Unknown'
        return name_after_check
        
    
    # In the face_recognize method, if there are only unknowns in the list 
    # of names, it will not print out anything.
    def face_recognize(self, model=None):
        video_capture = cv2.VideoCapture(0)
        if model == None:
            while True:
                # Grab a single frame of video
                ret, frame = video_capture.read()
            
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]
            
                # Only process every other frame of video to save time
                if self.process_this_frame:
                    # Find all the faces and face encodings in the current frame of video
                    self.face_locations = face_recognition.face_locations(rgb_small_frame)
                    self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
            
                    self.face_names = []
                    for face_encoding in self.face_encodings:        
                        self.face_names.append(convert(self.get_face_name(face_encoding)))
            
                self.process_this_frame = not self.process_this_frame
                for name in self.face_names:
                    if name != 'Unknown':
                        print(name)
        
                # Display the resulting image
                cv2.imshow('Video', frame)
            
                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        else:
            # one problem occurs when pass anything other than the model in 
            # the code. put in try catch rather than do a check type in 
            # if else. Update: do to the natural of the get_face_name_by_model
            # method the try catch will be put there instead of here. That try
            # catch will force any errors in to return 'Unknown', including 
            # ValueError, TypeError or IndexError
            while True:
                # Grab a single frame of video
                ret, frame = video_capture.read()
            
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]
            
                # Only process every other frame of video to save time
                if self.process_this_frame:
                    # Find all the faces and face encodings in the current frame of video
                    self.face_locations = face_recognition.face_locations(rgb_small_frame)
                    self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                    
                    self.face_names = []
                    for face_encoding in self.face_encodings:
                        self.face_names.append(self.get_face_name_by_model(face_encoding, model))
            
                self.process_this_frame = not self.process_this_frame
                for name in self.face_names:
                    # the frame that is not processed anything will be the one 
                    # that forced to print out result
                    if name != 'Unknown' and not self.process_this_frame:
                        print(name)

                # Display the resulting image
                cv2.imshow('Video', frame)
            
                # Hit 'q' on the keyboard to quit!
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                
        video_capture.release()
        cv2.destroyAllWindows()