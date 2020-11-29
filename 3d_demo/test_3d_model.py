import face_alignment
# uncomment 2 lines below if you want to test the manual part of deciding what had been correctly modeled
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import collections
from skimage import io
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
# for small dataset, you might want to use decision tree
from sklearn.tree import DecisionTreeClassifier
import cv2
from sklearn.decomposition import PCA
import PIL.Image


def png_to_jpg(path_to_image):
    rgba_image = PIL.Image.open(path_to_image)
    rgb_image = rgba_image.convert('RGB')
    new_name = path_to_image.rsplit('.', 1)[0] + '.jpg'
    rgb_image = rgb_image.save(new_name)



def normalize(list_to_norm, start_range, stop_range):
    if start_range > stop_range:
        print('Start point of normalized range is currently bigger then stop point. Attempt to swap them.')
        start_range, stop_range = stop_range, start_range
    length = stop_range - start_range
    min_num = min(list_to_norm)
    max_num = max(list_to_norm)
    normalized = []
    for num in list_to_norm:
        normalized.append(length * ((num - min_num) / (max_num - min_num)) + start_range)
    return normalized


def calculate_distance(lists_of_landmarks, start, stop):
    return float(np.linalg.norm(lists_of_landmarks[start] - lists_of_landmarks[stop], ord=2))


def calculate_jaw_angle(lists_of_landmarks):
    jaw_h = lists_of_landmarks[0] - lists_of_landmarks[3]
    jaw_w = lists_of_landmarks[13] - lists_of_landmarks[16]
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
    chin_1 = lists_of_landmarks[7] - lists_of_landmarks[8]
    chin_2 = lists_of_landmarks[8] - lists_of_landmarks[9]
    return float(np.dot(chin_1, chin_2) / (np.linalg.norm(chin_1) * np.linalg.norm(chin_2)))


def calculate_nose_angle(lists_of_landmarks):
    nose_1 = lists_of_landmarks[27] - lists_of_landmarks[30]
    nose_2 = lists_of_landmarks[33] - lists_of_landmarks[30]
    return float(np.dot(nose_1, nose_2) / (np.linalg.norm(nose_1) * np.linalg.norm(nose_2)))


def calculate_jawbone_angle(lists_of_landmarks):
    jawbone_1 = lists_of_landmarks[0] - lists_of_landmarks[1]
    jawbone_2 = lists_of_landmarks[17] - lists_of_landmarks[16]
    return float(np.dot(jawbone_1, jawbone_2) / (np.linalg.norm(jawbone_1) * np.linalg.norm(jawbone_2)))


def calculate_face(lists_of_landmarks, list_of_focused_points):
    number_cal = []
    for i in range(len(list_of_focused_points)):
        for j in range(i, len(list_of_focused_points)):
            number_cal.append(calculate_distance(lists_of_landmarks, list_of_focused_points[i], list_of_focused_points[j]))

    special_distance = [
        calculate_eye_width(lists_of_landmarks), calculate_eye_height(lists_of_landmarks),
        calculate_eye_brow_distance(lists_of_landmarks), calculate_upper_lip_height(lists_of_landmarks),
        calculate_lower_lip_height(lists_of_landmarks)
    ]
    special_distance = normalize(special_distance, 0, 1)
    special_angle = [calculate_jaw_angle(lists_of_landmarks), calculate_chin_angle(lists_of_landmarks),
                     calculate_nose_angle(lists_of_landmarks), calculate_jawbone_angle(lists_of_landmarks)]
    number_cal = normalize(number_cal, 0, 1) + special_distance + special_angle
    return number_cal


def load(trainset_path):
    train_dir = os.listdir(trainset_path)
    paths = []
    # Loop through each person in the training directory
    for person_type in train_dir:
        try:
            pix = os.listdir(trainset_path + person_type)
            # Loop through each training image for the current person
            for person_img in pix:
                try:
                    paths.append(trainset_path + person_type + '/' + person_img)
                except IndexError:
                    print('Cannot locate face. Skip this Image: ' + person_type + "/" + person_img)
        except NotADirectoryError:
            continue
    return paths


# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)

paths = load('dataset/testset/')
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
selected_points = [0, 3, 4, 7, 8, 9, 12, 13, 16, 17, 19, 21, 22, 24, 26, 27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39,
                   40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 52, 56, 57, 58, 61, 63, 67]

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
# code to substitute the code segment that had been commented above. Essentially going to get identical result
preds = []
label = pd.read_csv('dataset/testset/label.txt')
final_label = label.iloc[:, -1].values
for i in range(len(paths)):
    input_img = io.imread(paths[i])
    pred = fa.get_landmarks(input_img)[-1]
    preds.append(calculate_face(pred, selected_points))

final_table = pd.DataFrame(preds)
final_table['Label'] = final_label
with pd.ExcelWriter('dataset/testset/testset-3d.xlsx', 'openpyxl', mode='w') as wr:
    final_table.to_excel(wr, index=False)

#load trainset
crit_3d = pd.read_excel('dataset/test-3d.xlsx')

X_train = crit_3d.iloc[:, :-1].values
y_train = crit_3d.iloc[:, -1].values
# load testset
X_test = final_table.iloc[:, :-1].values
y_test = final_table.iloc[:, -1].values

clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.01)
clf = RandomForestClassifier(200, criterion='gini', max_features='auto', bootstrap=False,
                             class_weight='balanced', warm_start=False)
clf = SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr', probability=True)
clf.fit(X_train, y_train)


# after this write function to decide whether the highest probability class is good enough for a conclusion
# below is a proposed function
# name: is the content of the array return from predict function (which aligned with the highest probability class
# in the 2nd arg
# prob_each_class: is the array from predict_proba method return the probability of each class
# clf_class_name: list of class names of the classifier (label)
# thres: activation value, if prob > thres auto return name, otherwise go into detail abt the value
def name_decider(prob_each_class, clf_class_name, thres, actv_thres):
    # how this works get the index of the max probability -> index of class name
    # if n < 3: return name right away
    # if max_prob < actv_thres return unknown right away
    # if max_prob >= thres also return name right away
    # if max_prob - 2nd_max >= 2/n also return name
    # if more than 3 max pos return unknown
    # else return 3 possible name
    max_prob = float(prob_each_class.max())
    tmp = prob_each_class.copy()
    tmp.sort()
    second_max_prob = float(tmp[0][-2])
    third_max_prob = float(tmp[0][-3])
    n_class = prob_each_class.shape[1]
    if np.count_nonzero(prob_each_class == max_prob) == 1:
        name = clf_class_name[np.where(prob_each_class == max_prob)[1]][0]
        # the copy() method is crucial (similar to pass-by reference and pass-by value type of problem)
        if max_prob < actv_thres:
            return 'Unknown'
        if max_prob >= thres:
            return name
        if second_max_prob < actv_thres:
            return name
        if max_prob - second_max_prob >= 1 / n_class:
            return name
        else:
            return (name, clf_class_name[np.where(prob_each_class == second_max_prob)[1]][0])
    else:
        max_pos = np.where(prob_each_class == max_prob)[1]
        if len(max_pos) > 3:
            return 'Unknown'
        elif len(max_pos) == 3:
            max_pos = max_pos.tolist()
        elif len(max_pos) == 2:
            max_pos = max_pos.tolist()
            max_pos.append(np.where(prob_each_class == third_max_prob)[1].tolist()[0])

        return tuple(clf_class_name[i] for i in max_pos)


y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

final_predicts = []
for i in range(len(y_pred)):
    print(str(i) + ': ')
    tmp = name_decider(np.asarray([y_pred_proba[i]]), clf.classes_, 0.55, 0.4)
    try:
        print('Predicted: ' + tmp)
    except TypeError:
        for name in tmp:
            print('Predicted: ' + name)
    finally:
        final_predicts.append(tmp)
    print('True: ' + y_test[i])
    if y_test[i] == 'Unknown' and tmp != 'Unknown':
        print('Path to deflective: ' + paths[i])
    if y_test[i] != 'Unknown':
        print('Path to known image: ' + paths[i])
    print('--------------------------------------------------\n')

unknown_case = 0
known_case = 0
type_2 = 0
type_1 = 0
for predict, truth in zip(final_predicts, y_test):
    if truth == 'Unknown' and predict == 'Unknown':
        unknown_case += 1
    elif truth == 'Unknown' and predict != 'Unknown':
        unknown_case += 1
        type_2 += 1
    elif truth == predict:
        known_case += 1
    elif truth in predict:
        known_case += 1
    else:
        known_case += 1
        if truth != 'Nguyễn Ngọc Lâm':
            # font does not match between file system
            type_1 += 1





