import os
import cv2
import glob
import keras
import numpy as np
from random import shuffle
from collections import OrderedDict
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

owd = os.getcwd()

FULL_DIRECTORY_AFFWILD2 = '../data/aff-wild2/image/*/*'

IMG_ROWS = 48
IMG_COLS = 48
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.2
FEATURES_FOLDER = '../features/'


def image_preprocesing(path):
    """
        Converts an img to grayscale and resizes it to 48 x 48 pixels

        path: path to img
    """

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_ROWS, IMG_COLS))

    return img


def labels_preprocessing(emotion, classes):
    """
        Converts a label to categorical labels

        emotion: emotion labels (0, 1, 2 ...) (string)
        classes: total number of unique emotion labels (int)
    """

    _emotion = int(emotion)
    emotion = keras.utils.to_categorical(_emotion, classes)

    return emotion


def img_list_to_npy(lst):
    """
        Converts an list of img pixels to numpy

        list: list of img pixels (list)
    """

    _npy = np.array(lst)
    _npy = _npy.reshape(_npy.shape[0], IMG_ROWS, IMG_COLS, 1)
    npy = _npy.astype('float32')

    return npy


def save_as_npy(npy, npy_name):
    """
        Saves a numpy file to the specified features folder

        npy: list of img pixels (list)
        npy_name: name of numpy file (string)
    """

    if not os.path.exists(FEATURES_FOLDER):
        os.makedirs(FEATURES_FOLDER)

    np.save(FEATURES_FOLDER + npy_name, npy)


def _train_test_split(train_split, test_split, img_list, label_list):
    """
        train_split & test_split accepts value 0.0 - 1.0 (0% - 100%)
    """

    img_train = img_list[:int(len(img_list) * train_split)]
    label_train = label_list[:int(len(label_list) * train_split)]
    img_test = img_list[int(len(img_list) * (1.0 - test_split)):]
    label_test = label_list[int(len(label_list) * (1.0 - test_split)):]
    img_full = img_list[:int(len(img_list) * 1.0)]
    label_full = label_list[:int(len(label_list) * 1.0)]

    return img_train, label_train, img_test, label_test, img_full, label_full


def prepare_data(directory, classes, npy_name):
    """
        Converts data into a numpy format with the specified train and test
        split for CNN model baseline model training.

        directory: full path to dataset directory
        classes: total number of unique emotion labels (int)
        npy_name: name of numpy file (string)
    """

    _img_list = [[], [], [], [], [], [], []]
    _label_list = [[], [], [], [], [], [], []]
    _x_train, _y_train, _x_test, _y_test, _x_full, _y_full = [], [], [], [], \
                                                             [], []

    for path in glob.glob(directory):
        split_path = path.split(os.sep)
        _img_list[int(split_path[1])].append(path)
        _label_list[int(split_path[1])].append(split_path[1])

    for i in range(len(_img_list)):
        img_train, label_train, img_test, label_test, img_full, label_full = \
            _train_test_split(TRAIN_SPLIT, TEST_SPLIT, _img_list[i],
                              _label_list[i])

        for j in range(len(img_train)):
            _img_train = image_preprocesing(img_train[j])
            _x_train.append(_img_train)

        for j in range(len(label_train)):
            _label_train = labels_preprocessing(label_train[j], classes)
            _y_train.append(_label_train)

        for j in range(len(img_test)):
            _img_test = image_preprocesing(img_test[j])
            _x_test.append(_img_test)

        for j in range(len(label_test)):
            _label_train = labels_preprocessing(label_test[j], classes)
            _y_test.append(_label_train)

        for j in range(len(img_full)):
            _img_full = image_preprocesing(img_full[j])
            _x_full.append(_img_full)

        for j in range(len(label_full)):
            _label_full = labels_preprocessing(label_full[j], classes)
            _y_full.append(_label_full)

    _x_train = img_list_to_npy(_x_train)
    _x_test = img_list_to_npy(_x_test)
    _x_full = img_list_to_npy(_x_full)

    save_as_npy(_x_train, npy_name + "_train_images")
    save_as_npy(_y_train, npy_name + "_train_labels")
    save_as_npy(_x_test, npy_name + "_test_images")
    save_as_npy(_y_test, npy_name + "_test_labels")
    save_as_npy(_x_full, npy_name + "_full_images")
    save_as_npy(_y_full, npy_name + "_full_labels")


def _remove_duplicate(item):
    return list(OrderedDict.fromkeys(item))


def _prepare_affwild2_testdata(directory, classes, npy_name):
    """
        Converts affwild2 test set data into a numpy format..

        directory: full path to dataset directory
        classes: total number of unique emotion labels (int)
        npy_name: name of numpy file (string)
    """

    _img_list = [[], [], [], [], [], [], []]
    _label_list = [[], [], [], [], [], [], []]
    _x_full, _y_full = [], []

    for path in glob.glob(directory):
        split_path = path.split(os.sep)
        _img_list[int(split_path[4])].append(path)
        _label_list[int(split_path[4])].append(split_path[4])

    for i in range(len(_img_list)):
        img_full, label_full = _img_list[i], _label_list[i]

        for j in range(len(img_full)):
            _img_full = image_preprocesing(img_full[j])
            _x_full.append(_img_full)

        for j in range(len(label_full)):
            _label_full = labels_preprocessing(label_full[j], classes)
            _y_full.append(_label_full)

    _x_full = img_list_to_npy(_x_full)

    save_as_npy(_x_full, npy_name + "_full_images")
    save_as_npy(_y_full, npy_name + "_full_labels")


def prepare_affwild2(directory):
    """
        Converts affwild2 test set data into a numpy format..
    """

    _list = []
    for subdir, dirs, files in os.walk(directory):
        for _dir in dirs:
            dir_path = subdir
            _list.append(dir_path)

    dir_path_list = _remove_duplicate(_list)

    for i in range(len(dir_path_list)):
        if i == 0:
            pass
        else:
            __dir = dirpath_list[i] + "\*\*"
            _name_list = __dir.split(os.sep)
            name = __dir.split(os.sep)[len(_name_list) - 3]
            _prepare_affwild2_testdata(__dir, 7, str(name))


def merge_data(npy_files, merged_img_name, merged_label_name):
    """
        Merges two dataset numpy file together.

        npy_files: list of numpy files (list)
        merged_img_name: name of merged image numpy file (string)
        merged_label_name: name of merged image' labels numpy file (string)
    """

    merged_images, merged_labels = [], []

    for i in range(len(npy_files)):
        if "images" in npy_files[i]:
            print(npy_files[i])
            _images = np.load('../features/' + npy_files[i] + '.npy').tolist()
            merged_images.extend(_images)

        if "labels" in npy_files[i]:
            print(npy_files[i])
            _labels = np.load('../features/' + npy_files[i] + '.npy').tolist()
            merged_labels.extend(_labels)

    save_as_npy(merged_images, merged_img_name + '.npy')
    save_as_npy(merged_labels, merged_label_name + '.npy')


def main():
    """
        Run your sequence of program here.

        Use prepare_data() function to convert single dataset into .npy format.

    """

    # Prepare AFF-WILD2 TRAINING / VALIDATION DATA
    prepare_data(FULL_DIRECTORY_AFFWILD2, 7, "affwild2")

    # Prepare AFF-WILD2 TEST DATA
    prepare_affwild2('..\\data\\aff-wild2_test')


if __name__ == '__main__':
    main()
