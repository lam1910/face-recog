import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import os

PATH_TO_IMAGE_KNOWN = 'dataset/known_faces'
PATH_TO_IMAGE_UNKNOWN = 'dataset/unknown_faces'

# This is an example of running face recognition on a single image
# and drawing a box around each person that was identified.

# Load a sample picture and learn how to recognize it.
me_image = face_recognition.load_image_file(os.path.join(PATH_TO_IMAGE_KNOWN, 'me.jpg'))
me_face_encoding = face_recognition.face_encodings(me_image)[0]

# Load a second sample picture and learn how to recognize it.
# biden_image = face_recognition.load_image_file("biden.jpg")
# biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    me_face_encoding,
    #biden_face_encoding
]
known_face_names = [
    "Nguyen Ngoc Lam"
]

# Load an image with an unknown face
unknown_image = face_recognition.load_image_file(os.path.join(PATH_TO_IMAGE_UNKNOWN, 'IMG_0153.jpg'))

# Find all the faces and face encodings in the unknown image
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
# See http://pillow.readthedocs.io/ for more about PIL/Pillow
pil_image = Image.fromarray(unknown_image)
# Create a Pillow ImageDraw Draw instance to draw with
draw = ImageDraw.Draw(pil_image)

matches = []
face_distances = []
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    # See if the face is a match for the known face(s)
    matches.append(face_recognition.compare_faces(known_face_encodings, face_encoding))

    # If a match was found in known_face_encodings, just use the first one.
    # if True in matches:
    #     first_match_index = matches.index(True)
    #     name = known_face_names[first_match_index]

    # Or instead, use the known face with the smallest distance to the new face
    face_distances.append(face_recognition.face_distance(known_face_encodings, face_encoding))

names = ['Unknown'] * 10
best_match_index = np.argmin(face_distances)
if matches[best_match_index]:
    names[best_match_index] = known_face_names[0]

correct_color = (0, 255, 0)
incorrect_color = (255, 0, 0)
text_box = (0, 255, 0)

# Loop through each face found in the unknown image
for (top, right, bottom, left), name in zip(face_locations, names):
    # See if the face is a match for the known face(s)
    text_width, text_height = draw.textsize(name)
    # Draw a box around the face using the Pillow module
    if name == 'Unknown':
        draw.rectangle(((left, top), (right, bottom)), outline=incorrect_color)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=text_box,
                       outline=text_box)
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    else:
        draw.rectangle(((left, top), (right, bottom)), outline=correct_color)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=text_box,
                       outline=text_box)
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

# Remove the drawing library from memory as per the Pillow docs
del draw

# Display the resulting image
pil_image.show()