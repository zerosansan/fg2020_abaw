import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
import scipy.misc
import dlib
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pickle
import logging
from tqdm import tqdm
from keras.optimizers import SGD
from sklearn import preprocessing
from imutils import face_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Convolution2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from numpy.random import seed
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import precision_recall_fscore_support as score
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

owd = os.getcwd()

NUM_CORES = 8
NUM_CLASSES = 7
LABELS = ['Neutral', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']
BATCH_SIZE = 512
EPOCHS = 100
IMG_ROWS = 48
IMG_COLS = 48
SEED = 8
fit = True
seed(8)

CNN_LR = 0.001
CNN_DECAY = 1e-6
CNN_MOMENTUM = 0.9


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=24)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=18)
    plt.yticks(tick_marks, classes, fontsize=18)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=18
        )

    plt.ylabel('True label', fontsize=22)
    plt.xlabel('Predicted label', fontsize=22)
    plt.tight_layout()


def plot_classification_loss(history):
    plt.clf()

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'bo', label="Training loss")
    plt.plot(epochs, val_loss, 'b', label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_classification_acc(history):
    plt.clf()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label="Training accuracy")
    plt.plot(epochs, val_acc, 'b', label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def model_generate_CNN():
    logging.info("Generating CNN model....")

    model = Sequential()

    # 1st Conv layer
    model.add(Conv2D(32, (3, 3), activation='relu',
                     input_shape=(IMG_ROWS, IMG_COLS, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 2nd Conv layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd Conv layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Flattening
    model.add(Flatten())

    # Fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    # Output
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    sgd = SGD(lr=CNN_LR, decay=CNN_DECAY, momentum=CNN_MOMENTUM, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy']
                  )

    # model.summary()

    return model


def test_model():
    model = Sequential()

    model.add(
        Convolution2D(64, (3, 1), padding='same', input_shape=(48, 48, 1)))
    model.add(Convolution2D(64, (1, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, (3, 1), padding='same'))
    model.add(Convolution2D(128, (1, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, (3, 1), padding='same'))
    model.add(Convolution2D(256, (1, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(512, (3, 1), padding='same'))
    model.add(Convolution2D(512, (1, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(7))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # model.summary()

    return model


def train_CNN_K_Means(x_train, y_train, model_save_name):
    n_splits = 10
    i = 0
    conf_mat_avg = 0
    cv_scores = []
    conf_mat = []

    # model = model_generate_CNN()
    model = test_model()

    # define 10-fold cross validation test harness
    kfold = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=SEED
    )

    for train_index, test_index in tqdm(kfold.split(x_train, y_train),
                                        total=kfold.get_n_splits(),
                                        desc="k-fold"):
        i = i + 1

        print("Running Fold", i, "/", n_splits)
        X_train, X_test = x_train[train_index], x_train[test_index]
        Y_train, Y_test = y_train[train_index], y_train[test_index]

        model_checkpoint = ModelCheckpoint(
            "../model/CNN_model_" + model_save_name + ".h5",
            'val_accuracy',
            verbose=1,
            save_best_only=True
        )

        reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.5,
                                      patience=50, min_lr=0.0001)

        callbacks = [model_checkpoint, reduce_lr]

        history = model.fit(X_train, Y_train, epochs=EPOCHS, verbose=0,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_test, Y_test),
                            callbacks=callbacks)

        model.save_weights('../model/CNN_model_' + model_save_name + '_Last.h5')
        model.load_weights('../model/CNN_model_' + model_save_name + '.h5')

        # Save the model and the weights
        model_json = model.to_json()
        with open("../model/CNN_model_" + model_save_name + ".json", "w") \
                as json_file:
            json_file.write(model_json)

        # save the loss and accuracy data
        f = open('../model/CNN_history_' + model_save_name + '.pckl', 'wb')
        pickle.dump(history.history, f)
        f.close()

        # Evaluate model
        scores = model.evaluate(X_test, Y_test)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cv_scores.append(scores[1] * 100)

        print("%.2f%% (+/- %.2f%%)" % (np.mean(cv_scores), np.std(cv_scores)))

        y_prob = model.predict(X_test)
        y_classes = y_prob.argmax(axis=-1)
        y_pred = y_classes
        y_true = [0] * len(y_pred)

        for j in range(0, len(Y_test)):
            max_index = np.argmax(Y_test[j])
            y_true[j] = max_index

        conf_mat.append(confusion_matrix(y_true, y_pred,
                                         labels=range(NUM_CLASSES)))

    acc_score = np.mean(cv_scores)
    conf_mat = np.array(conf_mat)

    for i in range(len(conf_mat)):
        conf_mat_avg += conf_mat[i]

    conf_mat_avg = conf_mat_avg / 10

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(
        conf_mat_avg,
        classes=LABELS,
        normalize=True,
        title="Average Accuracy %: " + str(round(acc_score, 2))
    )

    plt.savefig("../results/train_" + str(model_save_name) + ".png")


def test_CNN_K_Means(x_test, y_test, model_save_name, test_set_name):
    conf_mat = []
    conf_mat_avg = 0
    cv_scores = []
    test_amount = 1

    # model = model_generate_CNN()
    model = test_model()

    model.load_weights('../model/CNN_model_' + model_save_name + '.h5')

    for i in range(test_amount):

        y_pred = model.predict_classes(x_test)
        y_true = [0] * len(y_pred)

        for j in range(0, len(y_test)):
            max_index = np.argmax(y_test[j])
            y_true[i] = max_index

        # Draw the confusion matrix
        conf_mat.append(confusion_matrix(y_true, y_pred,
                                         labels=range(NUM_CLASSES)))

        # Evaluate the model on the test set
        scores = model.evaluate(x_test, y_test)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        ac = float("{0:.3f}".format(scores[1] * 100))
        cv_scores.append(ac)

    acc_score_avg = np.mean(cv_scores)
    conf_mat = np.array(conf_mat)

    for i in range(len(conf_mat)):
        conf_mat_avg += conf_mat[i]

    conf_mat_avg = conf_mat_avg / test_amount

    # print(y_true)
    # print(y_pred)

    precision, recall, fscore, support = score(y_true, y_pred)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))

    precision_list = precision.tolist()
    recall_list = recall.tolist()
    fscore_list = fscore.tolist()

    fscore_avg = 0
    recall_avg = 0
    precision_avg = 0

    for i in range(len(fscore_list)):
        fscore_avg += fscore_list[i]

    for i in range(len(recall_list)):
        recall_avg += recall_list[i]

    for i in range(len(precision_list)):
        precision_avg += precision_list[i]

    fscore_avg = fscore_avg / len(fscore_list)
    recall_avg = recall_avg / len(recall_list)
    precision_avg = precision_avg / len(precision_list)

    print("AVG F1-SCORE: ", fscore_avg)
    print("AVG RECALL: ", recall_avg)
    print("AVG PRECISION: ", precision_avg)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(
        conf_mat_avg,
        classes=LABELS,
        normalize=True,
        title="Average Accuracy %: " + str(round(acc_score_avg, 2))
    )

    plt.savefig("../results/train_" + str(model_save_name) + "-test_" +
                str(test_set_name) + ".png")


def load_ABAW2020_test_files(directory):
    """
        Load all aff-wild2 test .npy files
    """

    images_list = []
    labels_list = []
    name_list = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = subdir + os.sep + file
            if "images" in file:
                images_list.append(filepath)
                _name = (os.path.splitext(file)[0]).replace("_full_images", "")
                name_list.append(_name)
            if "labels" in file:
                labels_list.append(filepath)

    return images_list, labels_list, name_list


def test_ABAW2020_CNN_K_Means(x_test, y_test, model_save_name, test_set_name):
    conf_mat = []
    conf_mat_avg = 0
    cv_scores = []
    test_amount = 1

    # model = model_generate_CNN()
    model = test_model()

    model.load_weights('../model/CNN_model_' + model_save_name + '.h5')

    for i in range(test_amount):

        y_pred = model.predict_classes(x_test)

    # print(y_true)
    # print(y_pred)

    prediction_filename = test_set_name + ".txt"
    pred_list = y_pred.tolist()

    os.chdir("../results")
    f = open(prediction_filename, "w+")
    f.write("Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise" + "\n")
    for i in range(len(pred_list)):
        f.write(str(pred_list[i]) + "\n")
    f.close()


def initialize_tensorflow(num_cores):
    """
        cpu - gpu configuration
        config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 56} )
        max: 1 gpu, 56 cpu
    """

    config = tf.ConfigProto(
        intra_op_parallelism_threads=num_cores,
        inter_op_parallelism_threads=num_cores,
        allow_soft_placement=True,
        device_count={'CPU': 1, 'GPU': 1}
    )

    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)


def perform_train_and_test():

    # DATASET loading
    # affwild2_train_images = np.load('../features/affwild2_train_images.npy')
    # affwild2_train_labels = np.load('../features/affwild2_train_labels.npy')
    # affwild2_val_images = np.load('../features/affwild2_test_images.npy')
    # affwild2_val_labels = np.load('../features/affwild2_test_labels.npy')

    # FG-2020 ABAW AFF-WILD2 training and validation
    # train_CNN_K_Means(affwild2_train_images, affwild2_train_labels, "affwild2")
    # test_CNN_K_Means(affwild2_val_images, affwild2_val_labels, "affwild2",
    #                  "affwild2")

    # FG-2020 ABAW AFF-WILD2 testing
    images_list, labels_list, name_list = load_ABAW2020_test_files(
        '../features/abaw2020_affwild_test_set')

    for i in range(len(images_list)):
        load_images = np.load(images_list[i])
        load_labels = np.load(labels_list[i])
        test_ABAW2020_CNN_K_Means(load_images, load_labels, "affwild2",
                                  name_list[i])


def main():
    initialize_tensorflow(NUM_CORES)

    if not os.path.exists('../results'):
        os.makedirs('../results')

    perform_train_and_test()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    main()
