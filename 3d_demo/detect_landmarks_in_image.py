import face_alignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
import collections
import cv2
import numpy as np
from sklearn.preprocessing import normalize


def calculate_jaw_height(lists_of_landmarks):
    return float(np.linalg.norm(lists_of_landmarks[0] - lists_of_landmarks[4], ord=2))


def calculate_jaw_circumference(lists_of_landmarks):
    return 2 * float(np.linalg.norm(lists_of_landmarks[9] - lists_of_landmarks[5], ord=2))


def calculate_jaw_angle(lists_of_landmarks):
    jaw_h = lists_of_landmarks[0] - lists_of_landmarks[4]
    jaw_w = lists_of_landmarks[9] - lists_of_landmarks[5]
    return float(np.dot(jaw_h, jaw_w) / (np.linalg.norm(jaw_h) * np.linalg.norm(jaw_w)))


def calculate_forehead_width(lists_of_landmarks):
    return float(np.linalg.norm(lists_of_landmarks[26] - lists_of_landmarks[17]))


def calculate_eyebrows_distance(lists_of_landmarks):
    return float(np.linalg.norm(lists_of_landmarks[21] - lists_of_landmarks[22]))


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


def calculate_nose_length(lists_of_landmarks):
    return float(np.linalg.norm(lists_of_landmarks[27] - lists_of_landmarks[31], ord=2))


def calculate_nose_width(lists_of_landmarks):
    return float(np.linalg.norm(lists_of_landmarks[31] - lists_of_landmarks[35], ord=2))


def calculate_lip_nostril_distance(lists_of_landmarks):
    return float(np.linalg.norm(lists_of_landmarks[33] - lists_of_landmarks[51], ord=2))


def calculate_mouth_width(lists_of_landmarks):
    return float(np.linalg.norm(lists_of_landmarks[48] - lists_of_landmarks[54], ord=2))


def calculate_upper_lip_height(lists_of_landmarks):
    ulip1_h = float(np.linalg.norm(lists_of_landmarks[50] - lists_of_landmarks[61]))
    ulip2_h = float(np.linalg.norm(lists_of_landmarks[52] - lists_of_landmarks[63]))
    return (ulip1_h + ulip2_h) / 2


def calculate_lower_lip_height(lists_of_landmarks):
    llip1_h = float(np.linalg.norm(lists_of_landmarks[56] - lists_of_landmarks[65]))
    llip2_h = float(np.linalg.norm(lists_of_landmarks[58] - lists_of_landmarks[67]))
    return (llip1_h + llip2_h) / 2


def calculate_lip_chin_distance(lists_of_landmarks):
    return float(np.linalg.norm(lists_of_landmarks[9] - lists_of_landmarks[57], ord=2))


def calculate_chin_angle(lists_of_landmarks):
    chin_1 = lists_of_landmarks[8] - lists_of_landmarks[9]
    chin_2 = lists_of_landmarks[9] - lists_of_landmarks[10]
    return float(np.dot(chin_1, chin_2) / (np.linalg.norm(chin_1) * np.linalg.norm(chin_2)))


def calculate_face(lists_of_landmarks):
    criteria = ['Jaw height', 'Jaw circumference', 'Forehead width', 'Eyebrows distance',
                'Eye width', 'Eye height', 'Eye-eyebrow distance', 'Nose length', 'Nose width',
                'Upper lip-nostril distance', 'Mouth width', 'Upper lip height', 'Lower lip height',
                'Lower lip-chin distance', 'Lower jawbone angle', 'Chin angle']
    number_cal = \
        [
            calculate_jaw_height(lists_of_landmarks), calculate_jaw_height(lists_of_landmarks),
            calculate_forehead_width(lists_of_landmarks), calculate_eyebrows_distance(lists_of_landmarks),
            calculate_eye_width(lists_of_landmarks), calculate_eye_height(lists_of_landmarks),
            calculate_eye_brow_distance(lists_of_landmarks), calculate_nose_length(lists_of_landmarks),
            calculate_nose_width(lists_of_landmarks), calculate_lip_nostril_distance(lists_of_landmarks),
            calculate_mouth_width(lists_of_landmarks), calculate_upper_lip_height(lists_of_landmarks),
            calculate_lower_lip_height(lists_of_landmarks), calculate_lip_chin_distance(lists_of_landmarks),
        ]
    number_cal = normalize([number_cal])
    number_cal = np.append(number_cal, calculate_jaw_angle(lists_of_landmarks))
    number_cal = np.append(number_cal, calculate_chin_angle(lists_of_landmarks))
    return dict(
        zip(
            criteria, number_cal
        )
    )


# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=True)

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

try:
    input_img = io.imread('../dataset/train_fol/Trần Thị Thùy Linh/118709585_685843948808260_6668964229337761510_n.jpg')
except FileNotFoundError:
    input_img = io.imread('dataset/train_fol/Trần Thị Thùy Linh/118709585_685843948808260_6668964229337761510_n.jpg')

# old image
preds = fa.get_landmarks(input_img)[-1]

# 2D-Plot
plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)

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

fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(input_img)

for pred_type in pred_types.values():
    ax.plot(preds[pred_type.slice, 0],
            preds[pred_type.slice, 1],
            color=pred_type.color, **plot_style)

ax.axis('off')

# 3D-Plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.scatter(preds[:, 0] * 1.2,
                  preds[:, 1],
                  preds[:, 2],
                  c='cyan',
                  alpha=1.0,
                  edgecolor='b')

for pred_type in pred_types.values():
    ax.plot3D(preds[pred_type.slice, 0] * 1.2,
              preds[pred_type.slice, 1],
              preds[pred_type.slice, 2], color='blue')

ax.view_init(elev=90., azim=90.)
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()
old_pic = calculate_face(preds)


# fresh image
preds = fa.get_landmarks(frame)[-1]

# 2D-Plot
plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)

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

fig = plt.figure(figsize=plt.figaspect(.5))
ax = fig.add_subplot(1, 2, 1)
ax.imshow(frame)

for pred_type in pred_types.values():
    ax.plot(preds[pred_type.slice, 0],
            preds[pred_type.slice, 1],
            color=pred_type.color, **plot_style)

ax.axis('off')

# 3D-Plot
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.scatter(preds[:, 0] * 1.2,
                  preds[:, 1],
                  preds[:, 2],
                  c='cyan',
                  alpha=1.0,
                  edgecolor='b')

for pred_type in pred_types.values():
    ax.plot3D(preds[pred_type.slice, 0] * 1.2,
              preds[pred_type.slice, 1],
              preds[pred_type.slice, 2], color='blue')

ax.view_init(elev=90., azim=90.)
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()
new_pic = calculate_face(preds)

# some camera produce the distance differently, for example, my dslr produces the distance calculated 2 times the
# laptop webcam
# attempt normalize, kind of success
# create an array
import pandas as pd
old_pic = pd.DataFrame([old_pic])
new_pic = pd.DataFrame([new_pic])
