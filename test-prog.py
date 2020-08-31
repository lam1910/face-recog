import face_recognition
import os

PATH_TO_IMAGE_KNOWN = 'dataset/known_faces'
PATH_TO_IMAGE_UNKNOWN = 'dataset/unknown_faces'

PERSON_CONCERNED = 'Tóc Tiên'

FULL_PATH_KNOWN = os.path.join(PATH_TO_IMAGE_KNOWN, PERSON_CONCERNED + '.jpg')
FULL_PATH_KNOWN_1 = os.path.join(PATH_TO_IMAGE_KNOWN, PERSON_CONCERNED + '_1.jpg')
FULL_PATH_KNOWN_2 = os.path.join(PATH_TO_IMAGE_KNOWN, PERSON_CONCERNED + '_2.jpg')
FULL_PATH_KNOWN_False = os.path.join(PATH_TO_IMAGE_KNOWN, 'Hằng Phương.jpg')
FULL_PATH_UNKNOWN = os.path.join(PATH_TO_IMAGE_UNKNOWN, PERSON_CONCERNED + ' (ca sĩ)', 'img_Tóc Tiên (ca sĩ)31.jpg')


known_image = face_recognition.load_image_file(FULL_PATH_KNOWN)
known_image_1 = face_recognition.load_image_file(FULL_PATH_KNOWN_1)
known_image_2 = face_recognition.load_image_file(FULL_PATH_KNOWN_2)
known_image_False = face_recognition.load_image_file(FULL_PATH_KNOWN_False)
unknown_image = face_recognition.load_image_file(FULL_PATH_UNKNOWN)

tt_encoding = face_recognition.face_encodings(known_image)[0]
tt_encoding_1 = face_recognition.face_encodings(known_image_1)[0]
tt_encoding_2 = face_recognition.face_encodings(known_image_2)[0]
tt_encoding_False = face_recognition.face_encodings(known_image_False)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([tt_encoding, tt_encoding_1, tt_encoding_2, tt_encoding_False],
                                         unknown_encoding, tolerance=0.45)
dis = face_recognition.face_distance([tt_encoding, tt_encoding_1, tt_encoding_2, tt_encoding_False], unknown_encoding)

if any(results[:-1]) and not results[-1]:
    print(PERSON_CONCERNED)
elif results[-1] is True and results[:-1].count(True) > 1:
    print(PERSON_CONCERNED)
    print('Case 2')
elif results[:-1].count(True) == 1:
    print('Inconclusive')
else:
    print('Not ' + PERSON_CONCERNED)
