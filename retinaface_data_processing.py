from utils_vector_function import calculate_distance, calculate_angle
from normalize_list import normalize
import numpy as np
import os
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm


cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}


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
                real_person = person.replace('_', ' ')
                names.append(real_person)
            except IndexError:
                print('Cannot locate face. Skip this Image: ' + person + "/" + person_img)

    return paths, names


PATH_TO_IMAGE = 'Pytorch_Retinaface/widerface_evaluate/real_face/'
paths, names = load(PATH_TO_IMAGE)

person_crucial_points = []
for path in paths:
    with open(path, 'r') as fptr:
        # read the third line
        for i in range(3):
            best_match = fptr.readline()
        best_match = best_match.split(' ')
        x1, y1 = best_match[4:6]
        x2, y2 = best_match[6:8]
        x3, y3 = best_match[8:10]
        x4, y4 = best_match[10:12]
        x5, y5 = best_match[12:14]
    person_crucial_points.append([(float(x1), float(y1)), (float(x2), float(y2)), (float(x3), float(y3)),
                                  (float(x4), float(y4)), (float(x5), float(y5))])

person_stats = []
distances = []
vectors = []
angles = []
for person in person_crucial_points:
    for i in range(len(person)):
        for j in range(i + 1, len(person)):
            distance = calculate_distance(person[i], person[j])
            distances.append(distance)
            vectors.append(np.asarray(person[j]) - np.asarray(person[i]))

    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            angle = calculate_angle(vectors[i], vectors[j])
            angles.append(angle)

    # compound distances and angles
    distances = normalize(distances, -1, 1)
    person_stat = distances + angles
    person_stats.append(person_stat)
    # reset distances, vectors and angles for next iter
    distances = []
    angles = []
    vectors = []

retinaface_people = pd.DataFrame(person_stats)
retinaface_people['Label'] = names

retinaface_people.to_excel('/home/lam/face-recog/dataset/wider_face_style_train/retinaface_real_people.xlsx',
                           index=False)

# test train model
X = retinaface_people.iloc[:, :-1].values
y = retinaface_people.iloc[:, -1].values
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import cv2


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    """ Old style model is stored with all names of parameters sharing common prefix 'module.' """
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


torch.set_grad_enabled(False)

cfg = cfg_mnet
net = RetinaFace(cfg=cfg, phase='test')
net = load_model(net, 'Pytorch_Retinaface/weights/mobilenet0.25_Final.pth', True)
net.eval()
cudnn.benchmark = True
device = torch.device("cpu")
net = net.to(device)

clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='sqrt', warm_start=False)
clf = AdaBoostClassifier(n_estimators=50, learning_rate=0.1)

clf.fit(X, y)
origin_size = True

video_capture = cv2.VideoCapture(0)
process_this_frame = True
while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # get the coordinate of the points
        img = np.float32(frame)

        # testing scale
        target_size = 1600
        max_size = 2150
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf, landms = net(img)  # forward pass
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > 0.02)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]

        # added lanmarks
        x1, y1 = landms[:, 0:2][0]
        x2, y2 = landms[:, 2:4][0]
        x3, y3 = landms[:, 4:6][0]
        x4, y4 = landms[:, 6:8][0]
        x5, y5 = landms[:, 8:10][0]

        conf = dets[0, -1]

        person = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)]
        person_stats = []
        distances = []
        vectors = []
        angles = []
        for i in range(len(person)):
            for j in range(i + 1, len(person)):
                distance = calculate_distance(person[i], person[j])
                distances.append(distance)
                vectors.append(np.asarray(person[j]) - np.asarray(person[i]))

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                angle = calculate_angle(vectors[i], vectors[j])
                angles.append(angle)

        # compound distances and angles
        distances = normalize(distances, -1, 1)
        person_stat = distances + angles

        face_names = clf.predict([person_stat])
        face_prob = clf.predict_proba([person_stat]).max()

    process_this_frame = not process_this_frame
    for name in face_names:
        # the frame that is not processed anything will be the one
        # that forced to print out result
        old_name = name
        if face_prob < 0.55:
            name = 'Unknown'

        if not process_this_frame:
            print(f'Output name: {name} (was {old_name}) with confidence of face {conf} and name confidence '
                  f'of {face_prob}')

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
