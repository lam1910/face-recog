import face_alignment
# uncomment 2 lines below if you want to test the manual part of deciding what had been correctly modeled
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import collections
from skimage import io
import numpy as np
from sklearn.preprocessing import normalize
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import cv2
from sklearn.decomposition import PCA


def calculate_distance(lists_of_landmarks, start, stop):
    return float(np.linalg.norm(lists_of_landmarks[start] - lists_of_landmarks[stop], ord=2))


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

    special_distance = [
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
pred_types = {
    'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
    'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
    'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
    'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
    'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
    'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
    'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
    'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
    'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
}
# -----------------------------------------------------------------------------------
# manually select which one the library can construct correctly the face
# preds = []
# final_label = []
# idx_to_check = -1
# # died at id = 9 (0-based)
#
# idx_to_check += 1
# input_img = io.imread(paths[idx_to_check])
# pred = fa.get_landmarks(input_img)[-1]
#
# # 2D-Plot
# plot_style = dict(marker='o',
#                   markersize=4,
#                   linestyle='-',
#                   lw=2)
#
# fig = plt.figure(figsize=plt.figaspect(.5))
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(input_img)
#
# for pred_type in pred_types.values():
#     ax.plot(pred[pred_type.slice, 0],
#             pred[pred_type.slice, 1],
#             color=pred_type.color, **plot_style)
#
# ax.axis('off')
#
# # 3D-Plot
# ax = fig.add_subplot(1, 2, 2, projection='3d')
# surf = ax.scatter(pred[:, 0] * 1.2,
#                   pred[:, 1],
#                   pred[:, 2],
#                   c='cyan',
#                   alpha=1.0,
#                   edgecolor='b')
#
# for pred_type in pred_types.values():
#     ax.plot3D(pred[pred_type.slice, 0] * 1.2,
#               pred[pred_type.slice, 1],
#               pred[pred_type.slice, 2], color='blue')
#
# ax.view_init(elev=90., azim=90.)
# ax.set_xlim(ax.get_xlim()[::-1])
# plt.show()
#
# preds.append(calculate_face(pred))
# final_label.append(names[idx_to_check])
# -----------------------------------------------------------------------------------

# id_to_append = [0, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15]
# code to substitute the code segment that had been commented above. Essentially going to get identical result
preds = []
final_label = []
for i in [0, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15]:
    input_img = io.imread(paths[i])
    pred = fa.get_landmarks(input_img)[-1]
    preds.append(calculate_face(pred))
    final_label.append(names[i])

# applying pca
pca = PCA(n_components=5, svd_solver='auto')
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
# max_feature = auto since having small number of features already
# min_sample_split is set to be 1 because of the size of the demo dataset. Idea dataset size: 5 pictures for each person
# be aware of the problem that some picture cannot correctly built the 3d model of the face, suggested that all the
# background of the pictures used to built the model be simple or not very colourful
# bootstrap set to false also to deal with small sample size (or rather a lot of pictures have to be removed as they
# cannot detect the face), and subsequently class_weight set to balanced_subsample. ith bigger sample size, you might
# want to set bootstrap back to true and class_weight to balanced to speed up the calculation
clf = RandomForestClassifier(500, criterion='gini', min_samples_split=1/11, max_features='auto', bootstrap=False,
                             class_weight='balanced_subsample', warm_start=False)
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

test_preds = fa.get_landmarks(frame)[-1]
new_pic = calculate_face(test_preds)
new_pic = np.array(new_pic)
new_pic = pca.transform([new_pic])

# get test pic
# test_img = io.imread('attendence_face/dataset/train_fol/Ma Chí Định/IMG_E0301.JPG')
# test_preds = fa.get_landmarks(test_img)[-1]
# test_pic = calculate_face(test_preds)
# predict at the point return likelihood of each classes instead of get the class name
# show out the probability of the picture is belong to each class
clf.predict_proba(new_pic)


# after this write function to decide whether the highest probability class is good enough for a conclusion
# below is a proposed function
# name: is the content of the array return from predict function (which aligned with the highest probability class
# in the 2nd arg
# prob_each_class: is the array from predict_proba method return the probability of each class
# clf_class_name: list of class names of the classifier (label)
# thres: activation value, if prob > thres auto return name, otherwise go into detail abt the value
def name_decider(name, prob_each_class, clf_class_name, thres):
    # how this works
    # if n < 3: return name right away
    # if max_prob >= thres also return name right away
    # if max_prob - 2nd_max >= 2/n also return name
    # if more than 3 max pos return unknown
    # else return 3 possible name

    max_prob = float(prob_each_class.max())
    # the copy() method is crucial (similar to pass-by reference and pass-by value type of problem)
    tmp = prob_each_class.copy()
    tmp.sort()
    second_max_prob = float(tmp[0][-2])
    third_max_prob = float(tmp[0][-3])
    n_class = prob_each_class.shape[1]
    if n_class < 3:
        return name
    if max_prob >= thres:
        return name
    if max_prob - second_max_prob >= 2 / n_class:
        return name

    # else case
    max_pos = np.where(prob_each_class == max_prob)[1]
    if len(max_pos) > 3:
        return 'Unknown'
    elif len(max_pos) == 3:
        max_pos = max_pos.tolist()
    elif len(max_pos) == 2:
        max_pos = max_pos.tolist()
        max_pos.append(np.where(prob_each_class == third_max_prob)[1].tolist()[0])
    elif len(max_pos) == 1:
        max_pos = max_pos.tolist()
        second_pos = np.where(prob_each_class == second_max_prob)[1]
        if len(second_pos) > 1:
            for i in [0, 1]:
                max_pos.append(second_pos.tolist()[i])
        else:
            max_pos.append(second_pos.tolist()[0])
            third_pos = np.where(prob_each_class == third_max_prob)[1]
            max_pos.append(third_pos.tolist()[0])

    return tuple(clf_class_name[i] for i in max_pos)


name_decider(clf.predict(new_pic)[0], clf.predict_proba(new_pic), clf.classes_, 0.4)
# clf.predict([list(test_pic.values())])
# -----------------------------------------------------------------------------------
