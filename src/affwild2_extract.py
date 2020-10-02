"""
    Script to convert affwild2 dataset into a 0-6 facial emotion labels folder
    convention as part of the data preparation process to convert to numpy
    format or merge with other datasets for baseline model training and testing.

    ## USAGE ##

    1. Within affwild2 dataset folder, extract the following zip files
    ("cropped_aligned.zip") and create a new folder called "annotation".
    2. Move "EXPR_Set" folder into the newly created "annotation" folder.
    3. Move this script into the root affwild2 folder and run the script.
"""

import os
import cv2
import csv
import zipfile
import rarfile
import shutil
import itertools
import time
import glob
from shutil import copyfile
from tqdm import tqdm
from collections import Counter
from distutils.dir_util import copy_tree

ROOT_DIR = os.getcwd()

NEUTRAL_FOLDER = ROOT_DIR + '\\labelled_image\\0'
ANGER_FOLDER = ROOT_DIR + '\\labelled_image\\1'
DISGUST_FOLDER = ROOT_DIR + '\\labelled_image\\2'
FEAR_FOLDER = ROOT_DIR + '\\labelled_image\\3'
HAPPY_FOLDER = ROOT_DIR + '\\labelled_image\\4'
SAD_FOLDER = ROOT_DIR + '\\labelled_image\\5'
SURPRISE_FOLDER = ROOT_DIR + '\\labelled_image\\6'

CROPPED_IMG_FOLDER = ROOT_DIR + '\\cropped_images'

IMAGE_FILEPATH = []
IMAGE_NAME = []
IMAGE_FILENAME = []
CROPPED_IMAGE_NAME = []
CROPPED_IMAGE_FILEPATH = []
LABEL_LIST = []
TESTING_FILE_LIST = []
TESTING_IMG_COUNT_LIST = []
TESTING_IMG_DIR_FILEPATH_LIST = []
IMAGE_LABEL_LIST = []
LABELLED_CROPPED_IMAGE_NAME = []


def _create_folder(name):
    if not os.path.exists(name):
        os.makedirs(name)


def get_images(directory, raw_img_filepath, raw_img_names, raw_img_filename):
    """
        Go through specified directory to look for all images' filepath.
    """

    os.chdir(directory)

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file

            path_list_len = len(filepath.split(os.sep)) - 1
            image_name = str(filepath.split(os.sep)[path_list_len - 1]) + '/' \
                + file
            image_name_no_ext = os.path.splitext(image_name)[0]
            image_file_name = str(filepath.split(os.sep)[path_list_len])

            raw_img_filepath.append(filepath)
            raw_img_names.append(image_name_no_ext)
            raw_img_filename.append(image_file_name)

    os.chdir(ROOT_DIR)


def get_labels(directory, label_list):
    """
        Go through all .txt files to look for all images labels and bounding
        boxes.
    """

    os.chdir(directory)

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            count = 1

            filepath = subdir + os.sep + file

            with open(filepath, "r") as fd:
                next(fd)
                for line in fd:
                    filepath_split = filepath.split("\\")
                    _folder_name = filepath_split[len(filepath_split) - 1]
                    folder_name_no_ext = os.path.splitext(_folder_name)[0]
                    line = line.strip()
                    label_list.append([str(folder_name_no_ext) + "/"
                                       + str(count).zfill(5), line])

                    count += 1

    os.chdir(ROOT_DIR)


def _create_dummy_annotation_file(directory, file_list, img_count_list):
    """
        Go through listed test file list and create annotation file for
        each files according to the format of given training and validation
        annotation files.
    """

    os.chdir(directory)

    _create_folder('Testing_Set')

    os.chdir('Testing_Set')

    for i in range(len(file_list)):
        f = open(file_list[i] + ".txt", "w+")
        f.write("Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise" + "\n")
        f.close()

    img_count_list_index = 0
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if img_count_list_index == len(img_count_list):
                pass
            else:
                filepath = subdir + os.sep + file

                with open(filepath, "a") as f:
                    for i in range(img_count_list[img_count_list_index]):
                        f.write("0" + "\n")

                img_count_list_index += 1

    os.chdir(ROOT_DIR)


def _get_test_img_dir_list(directory, file_list):
    """
        Go through specified directory to look for test images
    """

    os.chdir(directory)

    img_dir_list = []
    for subdir, dirs, files in os.walk(directory):
        for _dir in dirs:
            dir_path = subdir + _dir
            if _dir in file_list:
                img_dir_list.append(dir_path)
    os.chdir(ROOT_DIR)

    return img_dir_list


def _get_file_count_in_dir_list(dir_list):
    """
        Count number of images in a directory, for creation of
        dummy annotation labels.
    """

    dir_file_count_list = []
    for i in range(len(dir_list)):
        _dir_files = os.listdir(dir_list[i])
        file_cnt = len(_dir_files)
        dir_file_count_list.append(file_cnt)

    return dir_file_count_list


def setup_test_files(file, file_list, testing_img_dir_filepath_list):
    """
        Setup annotation files for test set and create dummy expression value
        of "0" for each image.
    """

    with open(file, "r") as fd:
        next(fd)
        for line in fd:
            line = line.strip()
            file_list.append(line)

    test_img_dir_list = _get_test_img_dir_list(ROOT_DIR + '\\cropped_aligned\\',
                                               file_list)
    testing_img_dir_filepath_list.extend(test_img_dir_list)
    test_img_dir_file_count_list = _get_file_count_in_dir_list(
        test_img_dir_list)
    _create_dummy_annotation_file(ROOT_DIR + '\\annotation\\EXPR_Set',
                                  file_list, test_img_dir_file_count_list)


def match_image_with_labels(img_name_list, img_filepath_list,
                            label_list, img_label_list):
    """
        Match image according to its emotion labels.
    """

    for i in tqdm(range(len(img_name_list))):
        img_name = img_name_list[i]
        for j in range(len(label_list)):
            label_name = label_list[j][0]
            label_cat = label_list[j][1]
            if img_name == label_name:
                new_img_name = img_name_list[i].replace('/', '_') + ".jpg"
                img_label_list.append([new_img_name, img_filepath_list[i],
                                       int(label_cat)])
                break


def arrange_files():
    """
        Using the folder conventions of 0-6 according to basic emotion labels.
        The images will be organized into these folders.
    """

    _create_folder('labelled_image')
    os.chdir('labelled_image')

    _create_folder('0')
    _create_folder('1')
    _create_folder('2')
    _create_folder('3')
    _create_folder('4')
    _create_folder('5')
    _create_folder('6')

    os.chdir(ROOT_DIR)

    for i in tqdm(range(len(IMAGE_LABEL_LIST))):
        filename = IMAGE_LABEL_LIST[i][0]
        filepath = IMAGE_LABEL_LIST[i][1]
        label = IMAGE_LABEL_LIST[i][2]
        if label == 0:  # neutral
            shutil.copy(filepath, os.path.join(NEUTRAL_FOLDER, filename))
        if label == 1:  # angry
            shutil.copy(filepath, os.path.join(ANGER_FOLDER, filename))
        if label == 2:  # disgust
            shutil.copy(filepath, os.path.join(DISGUST_FOLDER, filename))
        if label == 3:  # fear
            shutil.copy(filepath, os.path.join(FEAR_FOLDER, filename))
        if label == 4:  # happy
            shutil.copy(filepath, os.path.join(HAPPY_FOLDER, filename))
        if label == 5:  # sad
            shutil.copy(filepath, os.path.join(SAD_FOLDER, filename))
        if label == 6:  # surprise
            shutil.copy(filepath, os.path.join(SURPRISE_FOLDER, filename))


def _copy_file_to_another_dir(from_dir, to_dir):
    """
        Used to copy all .jpg files from a directory into another directory.
    """
    for file in glob.iglob(os.path.join(from_dir, "*.JPG")):
        shutil.copy(file, to_dir)


def arrange_test_files(directory, test_img_dir_name_list,
                       test_img_dir_path_list):
    """
        Using the folder conventions of 0-6 according to basic emotion labels.
        All test images will be arranged in neutral folder.
    """

    _create_folder('labelled_test_image')
    os.chdir('labelled_test_image')

    cwd = os.getcwd()

    for i in tqdm(range(len(test_img_dir_name_list))):
        _create_folder(test_img_dir_name_list[i])
        os.chdir(test_img_dir_name_list[i])
        _create_folder('0')
        _create_folder('1')
        _create_folder('2')
        _create_folder('3')
        _create_folder('4')
        _create_folder('5')
        _create_folder('6')

        os.chdir(cwd)

    img_list = []
    for i in tqdm(range(len(test_img_dir_path_list))):
        for subdir, dirs, files in os.walk(test_img_dir_path_list[i]):
            for file in files:
                filepath = subdir + os.sep + file
                img_list.append(filepath)

    dir_path_list = []
    for subdir, dirs, files in os.walk(directory):
        for _dir in dirs:
            dir_path = subdir + os.sep + _dir
            dir_path_list.append(dir_path)

    for i in tqdm(range(len(dir_path_list))):
        dir_path_split = dir_path_list[i].split("\\")
        dir_path_dirname = dir_path_split[len(dir_path_split) - 1]

        for j in tqdm(range(len(img_list))):
            filepath_split = img_list[j].split("\\")
            dir_name = filepath_split[len(filepath_split) - 2]

            if dir_path_dirname == dir_name:
                shutil.copy(img_list[j], dir_path_list[i] + "\\0")


def _setup_train_val_set():
    """
        Runs all functions to extract training and validation sets.
    """

    # Get all images and labels for both training and validation sets
    print("Running get_images()...")
    get_images(ROOT_DIR + '\\cropped_aligned\\', IMAGE_FILEPATH, IMAGE_NAME,
               IMAGE_FILENAME)
    print("Running get_labels() for training set...")
    get_labels(ROOT_DIR + '\\annotation\\EXPR_Set\\Training_Set', LABEL_LIST)
    print("Running get_labels() for Validation set...")
    get_labels(ROOT_DIR + '\\annotation\\EXPR_Set\\Validation_Set', LABEL_LIST)

    # Match all images with its specified labels
    print("Running match_image_with_labels()...")
    match_image_with_labels(IMAGE_NAME, IMAGE_FILEPATH, LABEL_LIST,
                            IMAGE_LABEL_LIST)

    # Arrange images into {0-6} folders according to its expression labels
    print("Running arrange_files()...")
    arrange_files()


def _setup_test_set():
    """
        Runs all functions to extract test sets.
        Note that this test set contains dummy values (all of which are neutral)
    """

    # Get all test set images and assign dummy neutral labels.
    print("Setting up dummy annotation files for test set...")
    setup_test_files(ROOT_DIR + '\\expression_test_set.txt', TESTING_FILE_LIST,
                     TESTING_IMG_DIR_FILEPATH_LIST)

    # Copy images of all test set folder to neutral folder.
    print("Running arrange_files()...")
    arrange_test_files(ROOT_DIR + "\\labelled_test_image", TESTING_FILE_LIST,
                       TESTING_IMG_DIR_FILEPATH_LIST)


def main():
    start = time.process_time()

    # TRAINING SET extract #
    # _setup_train_val_set()

    # TESTING SET extract #
    _setup_test_set()

    print("TIME ELAPSED:", time.process_time() - start)


if __name__ == '__main__':
    main()
