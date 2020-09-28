import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
from skimage import io
import numpy as np
from sklearn.preprocessing import normalize
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import cv2
from sklearn.decomposition import PCA


def calculate_distance(lists_of_landmarks, i, j):
    return float(np.linalg.norm(lists_of_landmarks[i] - lists_of_landmarks[j], ord=2))


def calculate_jaw_angle(lists_of_landmarks):
    jaw_h = lists_of_landmarks[0] - lists_of_landmarks[4]
    jaw_w = lists_of_landmarks[9] - lists_of_landmarks[5]
    return float(np.dot(jaw_h, jaw_w) / (np.linalg.norm(jaw_h) * np.linalg.norm(jaw_w)))


def calculate_eye_width(lists_of_landmarks):
    eye1_w = float(np.linalg.norm(lists_of_landmarks[36] - lists_of_landmarks[39]))
    eye2_w = float(np.linalg.norm(lists_of_landmarks[42] - lists_of_landmarks[45]))
    return (eye1_w + eye2_w) / 2


def calculate_eye_height(lists_of_landmarks):
    eye1_h_1 = float(np.linalg.norm(lists_of_landmarks[37] - lists_of_landmarks[41]))
    eye1_h_2 = float(np.linalg.norm(lists_of_landmarks[38] - lists_of_landmarks[40]))
    eye2_h_1 = float(np.linalg.norm(lists_of_landmarks[43] - lists_of_landmarks[47]))
    eye2_h_2 = float(np.linalg.norm(lists_of_landmarks[44] - lists_of_landmarks[46]))
    return (max([eye1_h_1, eye1_h_2]) + max([eye2_h_1, eye2_h_2])) / 2


def calculate_eye_brow_distance(lists_of_landmarks):
    eye1_brow_1 = float(np.linalg.norm(lists_of_landmarks[37] - lists_of_landmarks[19]))
    eye1_brow_2 = float(np.linalg.norm(lists_of_landmarks[38] - lists_of_landmarks[19]))
    eye2_brow_1 = float(np.linalg.norm(lists_of_landmarks[43] - lists_of_landmarks[24]))
    eye2_brow_2 = float(np.linalg.norm(lists_of_landmarks[44] - lists_of_landmarks[24]))
    return ((eye1_brow_1 + eye1_brow_2) / 2 + (eye2_brow_1 + eye2_brow_2) / 2) / 2


def calculate_upper_lip_height(lists_of_landmarks):
    ulip1_h = float(np.linalg.norm(lists_of_landmarks[50] - lists_of_landmarks[61]))
    ulip2_h = float(np.linalg.norm(lists_of_landmarks[52] - lists_of_landmarks[63]))
    return (ulip1_h + ulip2_h) / 2


def calculate_lower_lip_height(lists_of_landmarks):
    llip1_h = float(np.linalg.norm(lists_of_landmarks[56] - lists_of_landmarks[65]))
    llip2_h = float(np.linalg.norm(lists_of_landmarks[58] - lists_of_landmarks[67]))
    return (llip1_h + llip2_h) / 2


def calculate_chin_angle(lists_of_landmarks):
    chin_1 = lists_of_landmarks[8] - lists_of_landmarks[9]
    chin_2 = lists_of_landmarks[9] - lists_of_landmarks[10]
    return float(np.dot(chin_1, chin_2) / (np.linalg.norm(chin_1) * np.linalg.norm(chin_2)))


def calculate_nose_angle(lists_of_landmarks):
    nose_1 = lists_of_landmarks[27] - lists_of_landmarks[30]
    nose_2 = lists_of_landmarks[33] - lists_of_landmarks[30]
    return float(np.dot(nose_1, nose_2) / (np.linalg.norm(nose_1) * np.linalg.norm(nose_2)))


def calculate_jawbone_angle(lists_of_landmarks):
    jawbone_1 = lists_of_landmarks[0] - lists_of_landmarks[1]
    jawbone_2 = lists_of_landmarks[17] - lists_of_landmarks[16]
    return float(np.dot(jawbone_1, jawbone_2) / (np.linalg.norm(jawbone_1) * np.linalg.norm(jawbone_2)))


def calculate_face(lists_of_landmarks):
    number_cal = []
    for i in range(len(lists_of_landmarks)):
        for j in range(i, len(lists_of_landmarks)):
            number_cal.append(calculate_distance(lists_of_landmarks, i, j))

    special_distance = \
        [
            calculate_eye_width(lists_of_landmarks), calculate_eye_height(lists_of_landmarks),
            calculate_eye_brow_distance(lists_of_landmarks), calculate_upper_lip_height(lists_of_landmarks),
            calculate_lower_lip_height(lists_of_landmarks), calculate_jaw_angle(lists_of_landmarks),
            calculate_chin_angle(lists_of_landmarks), calculate_nose_angle(lists_of_landmarks),
            calculate_jawbone_angle(lists_of_landmarks)
        ]
    number_cal = number_cal + special_distance
    return normalize([number_cal]).tolist()[0]


def load(trainset_path):
    train_dir = os.listdir(trainset_path)
    paths = []
    names = []
    # Loop through each person in the training directory
    for person in train_dir:
        pix = os.listdir(trainset_path + person)
        # Loop through each training image for the current person
        for person_img in pix:
            try:
                paths.append(trainset_path + person + '/' + person_img)
                names.append(person)
            except IndexError:
                print('Cannot locate face. Skip this Image: ' + person + "/" + person_img)

    return paths, names


# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)

paths, names = load('dataset/train_fol/')
pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }
preds = []
final_label = []
idx_to_check = -1
# died at id = 9 (0-based)

idx_to_check += 1
input_img = io.imread(paths[idx_to_check])
pred = fa.get_landmarks(input_img)[-1]

# 2D-Plot
plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)

fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(input_img)

for pred_type in pred_types.values():
    ax.plot(pred[pred_type.slice, 0],
            pred[pred_type.slice, 1],
            color=pred_type.color, **plot_style)

ax.axis('off')

# 3D-Plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.scatter(pred[:, 0] * 1.2,
                  pred[:, 1],
                  pred[:, 2],
                  c='cyan',
                  alpha=1.0,
                  edgecolor='b')

for pred_type in pred_types.values():
    ax.plot3D(pred[pred_type.slice, 0] * 1.2,
              pred[pred_type.slice, 1],
              pred[pred_type.slice, 2], color='blue')

ax.view_init(elev=90., azim=90.)
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()

preds.append(calculate_face(pred))
final_label.append(names[idx_to_check])

# id_to_append = [0, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15]

# applying pca
pca = PCA(5, svd_solver='auto')
preds = pca.fit_transform(preds)

# final_table = pd.DataFrame(preds)
# final_table['Label'] = final_label
# # checkpoint at id = 8
# with pd.ExcelWriter('dataset/test-3d.xlsx', 'openpyxl', mode='a') as wr:
#     final_table.to_excel(wr, index=False)

# get trainset
crit_3d = pd.read_excel('dataset/test-3d.xlsx')

X = crit_3d.iloc[:, :-1].values
y = crit_3d.iloc[:, -1].values

# classifier
clf = RandomForestClassifier(20, random_state=0, warm_start=True)

clf.fit(X, y)

# -----------------------------------------------------------------------------------
# get a new picture to recognize
# define a video capture object
vid = cv2.VideoCapture(0)

while True:

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object, convert the color channel of the last pic
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

preds = fa.get_landmarks(frame)[-1]
new_pic = calculate_face(preds)
new_pic = pca.transform([new_pic])

# get test pic
# test_img = io.imread('attendence_face/dataset/train_fol/Ma Chí Định/IMG_E0301.JPG')
# test_preds = fa.get_landmarks(test_img)[-1]
# test_pic = calculate_face(test_preds)
# predict
clf.predict(new_pic)
# clf.predict([list(test_pic.values())])
# -----------------------------------------------------------------------------------
