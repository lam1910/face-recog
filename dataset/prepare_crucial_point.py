import os
import face_recognition


def get_face_location(img_path):
    unknown_image = face_recognition.load_image_file(os.path.join(img_path))
    face_locations = face_recognition.face_locations(unknown_image)
    try:
        a = face_locations[3]
    except IndexError:
        a = 0
        face_locations = face_locations[0]
    finally:
        del a
        return [
            face_locations[3], face_locations[0], face_locations[1] - face_locations[3],
            face_locations[2] - face_locations[0]
        ]


def get_face_locations_folder(folder_path):
    if folder_path[-1] != '/':
        folder_path += '/'
    train_dir = os.listdir(folder_path)
    locations = []
    filepaths = []
    # Loop through each person in the training directory
    for person in train_dir:
        try:
            pix = os.listdir(folder_path + person)
            # Loop through each training image for the current person
            for person_img in pix:
                try:
                    locations.append(get_face_location(folder_path + person + '/' + person_img))
                    filepaths.append(folder_path + person + '/' + person_img)
                except IndexError:
                    print('Cannot locate face. Skip this Image: ' + person + "/" + person_img)
        except NotADirectoryError:
            print('This is a file. Skip this.')

    return locations, filepaths


def to_txt(filename, data, src_paths):
    with open(filename, 'w') as ptr:
        for datum, src_path in zip(data, src_paths):
            ptr.write('# ' + src_path + '\n')
            for i in range(len(datum)):
                num = datum[i]
                if i != len(datum) - 1:
                    ptr.write(str(num) + ' ')
                else:
                    ptr.write(str(num) + '\n')
