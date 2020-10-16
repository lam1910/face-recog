import numpy as np


def calculate_distance(point1=None, point2=None, x1=0, y1=0, x2=0, y2=0):
    # distance between 2 points
    if (type(point1) == list or type(point1) == tuple) and (type(point2) == list or type(point2) == tuple):
        point1 = np.asarray(point1)
        point2 = np.asarray(point2)
    elif isinstance(point1, np.ndarray) and isinstance(point2, np.ndarray):
        pass
    else:
        point1 = (x1, y1)
        point2 = (x2, y2)
    return float(np.linalg.norm(point1 - point2, ord=2))


def calculate_length(vec=None, point1=None, point2=None, x1=0, y1=0, x2=0, y2=0, reverse=False):
    if (type(vec) == list or type(vec) == tuple) and (type(vec) == list or type(vec) == tuple):
        vec = np.asarray(vec)
    elif isinstance(vec, np.ndarray):
        pass
    else:
        if (type(point1) == list or type(point1) == tuple) and (type(point2) == list or type(point2) == tuple):
            point1 = np.asarray(point1)
            point2 = np.asarray(point2)
        elif isinstance(point1, np.ndarray) and isinstance(point2, np.ndarray):
            pass
        else:
            point1 = (x1, y1)
            point2 = (x2, y2)
        if not reverse:
            vec = point2 - point1
        else:
            vec = point1 - point2
    return float(np.linalg.norm(vec, ord=2))


def calculate_angle(vec1, vec2):
    # angle between 2 vec. return cos of the angle
    if (type(vec1) == list or type(vec1) == tuple) and (type(vec2) == list or type(vec2) == tuple):
        vec1 = np.asarray(vec1)
        vec2 = np.asarray(vec2)
    elif isinstance(vec1, np.ndarray) and isinstance(vec2, np.ndarray):
        pass
    else:
        print('Cannot recognized the parameters')
        # return cos = 0 meaning is cannot be close
        return 0

    return float(np.dot(vec1, vec2)) / (calculate_length(vec1) * calculate_length(vec2))

