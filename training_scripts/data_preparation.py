import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pickle
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
# from keras.utils import to_categorical
from src.utilFunctions import append_or_write
from src.phonemeMap import dic_pho_label
from src.phonemeMap import dic_pho_label_teacher_student

import logging
logging.basicConfig(level=logging.INFO)


def load_data(filename_labels_train_validation_set):

    # load training and validation data
    with open(filename_labels_train_validation_set, 'rb') as f:
        Y_train_validation = pickle.load(f)

    # this is the filename indices
    indices_features = range(len(Y_train_validation))

    indices_train, indices_validation, Y_train, Y_validation    = \
        train_test_split(indices_features, Y_train_validation, test_size=0.1, stratify=Y_train_validation)

    return indices_train, Y_train, \
           indices_validation, Y_validation, \
           indices_features, Y_train_validation


def load_data_embedding(filename_feature, filename_list_key, filename_scaler):
    """
    load data for the RNN phone embedding model
    :param filename_feature:
    :param filename_list_key:
    :return:
    """
    list_feature = pickle.load(open(filename_feature, 'rb'))
    list_key = pickle.load(open(filename_list_key, 'rb'))
    scaler = pickle.load(open(filename_scaler, 'rb'))

    # flatten the feature and label list
    list_feature_flatten = []
    list_key_flatten = []
    for ii in range(len(list_feature)):
        list_feature_flatten += list_feature[ii]
        list_key_flatten += [list_key[ii]]*len(list_feature[ii])

    # encode the label to categorical
    le = preprocessing.LabelEncoder()
    le.fit(list_key_flatten)
    label_integer = le.transform(list_key_flatten)

    # # train validation dataset split
    # X_train, X_val, y_train, y_val = train_test_split(list_feature_flatten, label_integer, test_size=0.1, stratify=label_integer)
    #
    # y_train = to_categorical(y_train)
    # y_val = to_categorical(y_val)

    return list_feature_flatten, label_integer, le, scaler


def featureFlatten(list_feature, list_key, data_str='_teacher'):
    """flatten the feature list"""
    list_feature_flatten = []
    list_key_flatten = []
    for ii in range(len(list_feature)):
        if list_key[ii] == u'?' or list_key[ii] == u'sil':
            pass
        else:
            list_feature_flatten += list_feature[ii]
            list_key_flatten += [list_key[ii] + data_str] * len(list_feature[ii])

    return list_feature_flatten, list_key_flatten


def load_data_embedding_teacher_student(filename_feature_teacher,
                                        filename_list_key_teacher,
                                        filename_feature_student,
                                        filename_list_key_student,
                                        filename_scaler):
    """
    load data for the RNN phone embedding model
    """
    list_feature_teacher = pickle.load(open(filename_feature_teacher, 'rb'))
    list_key_teacher = pickle.load(open(filename_list_key_teacher, 'rb'))
    list_feature_student = pickle.load(open(filename_feature_student, 'rb'))
    list_key_student = pickle.load(open(filename_list_key_student, 'rb'))
    scaler = pickle.load(open(filename_scaler, 'rb'))

    # flatten the feature and label list
    list_feature_flatten_teacher, list_key_flatten_teacher = \
        featureFlatten(list_feature_teacher, list_key_teacher, '_teacher')
    list_feature_flatten_student, list_key_flatten_student = \
        featureFlatten(list_feature_student, list_key_student, '_student')

    list_feature_flatten = list_feature_flatten_teacher + list_feature_flatten_student
    list_key_flatten = list_key_flatten_teacher + list_key_flatten_student

    # encode the label to integer
    le = preprocessing.LabelEncoder()
    le.fit(list_key_flatten)
    label_integer = le.transform(list_key_flatten)

    return list_feature_flatten, label_integer, le, scaler


def load_data_embedding_all(filename_feature_teacher_train,
                            filename_feature_teacher_val,
                            filename_feature_teacher_test,
                            filename_list_key_teacher,
                            filename_feature_student_train,
                            filename_feature_student_val,
                            filename_feature_student_test,
                            filename_list_key_student,
                            filename_feature_student_extra_test,
                            filename_list_key_extra_student,
                            filename_scaler):
    """
    load all data
    """
    list_feature_teacher_train = pickle.load(open(filename_feature_teacher_train, 'rb'))
    list_feature_teacher_val = pickle.load(open(filename_feature_teacher_val, 'rb'))
    list_feature_teacher_test = pickle.load(open(filename_feature_teacher_test, 'rb'))
    list_key_teacher = pickle.load(open(filename_list_key_teacher, 'rb'))
    list_feature_student_train = pickle.load(open(filename_feature_student_train, 'rb'))
    list_feature_student_val = pickle.load(open(filename_feature_student_val, 'rb'))
    list_feature_student_test = pickle.load(open(filename_feature_student_test, 'rb'))
    list_key_student = pickle.load(open(filename_list_key_student, 'rb'))
    list_feature_student_extra_test = pickle.load(open(filename_feature_student_extra_test, 'rb'))
    list_key_extra_student = pickle.load(open(filename_list_key_extra_student, 'rb'))
    scaler = pickle.load(open(filename_scaler, 'rb'))

    # flatten the feature and label list
    list_feature_flatten_teacher_train, list_key_flatten_teacher_train = \
        featureFlatten(list_feature_teacher_train, list_key_teacher, '_teacher')
    list_feature_flatten_teacher_val, list_key_flatten_teacher_val = \
        featureFlatten(list_feature_teacher_val, list_key_teacher, '_teacher')
    list_feature_flatten_teacher_test, list_key_flatten_teacher_test = \
        featureFlatten(list_feature_teacher_test, list_key_teacher, '_teacher')
    list_feature_flatten_student_train, list_key_flatten_student_train = \
        featureFlatten(list_feature_student_train, list_key_student, '_student')
    list_feature_flatten_student_val, list_key_flatten_student_val = \
        featureFlatten(list_feature_student_val, list_key_student, '_student')
    list_feature_flatten_student_test, list_key_flatten_student_test = \
        featureFlatten(list_feature_student_test, list_key_student, '_student')
    list_feature_flatten_student_extra_test, list_key_flatten_student_extra_test = \
        featureFlatten(list_feature_student_extra_test, list_key_extra_student, '_extra_test')

    list_feature_flatten = list_feature_flatten_teacher_train + list_feature_flatten_teacher_val + \
                           list_feature_flatten_teacher_test + list_feature_flatten_student_train + \
                           list_feature_flatten_student_val + list_feature_flatten_student_test + \
                           list_feature_flatten_student_extra_test
    list_key_flatten = list_key_flatten_teacher_train + list_key_flatten_teacher_val + \
                       list_key_flatten_teacher_test + list_key_flatten_student_train + \
                       list_key_flatten_student_val + list_key_flatten_student_test + \
                       list_key_flatten_student_extra_test

    return list_feature_flatten, list_key_flatten, scaler


def feature_replication(list_feature, list_key, scaler):
    from src.audio_preprocessing import _nbf_2D
    # flatten the feature and label list
    list_feature_flatten = []
    labels = []
    for ii in range(len(list_feature)):
        list_feature_flatten += list_feature[ii]
        labels += [dic_pho_label[list_key[ii]]] * len(list_feature[ii])

    logger = logging.getLogger(__name__)
    logger.info('collect feature...')

    array_feature_replicated = []
    array_labels = []
    for ii in range(len(list_feature_flatten)):
        # pad the phn feature if < 7
        if list_feature_flatten[ii].shape[0] < 7:
            feature_phn = np.pad(list_feature_flatten[ii],
                                 ((0, 7 - list_feature_flatten[ii].shape[0]), (0, 0)),
                                 'edge')
        else:
            feature_phn = list_feature_flatten[ii]

        array_feature_replicated.append(_nbf_2D(scaler.transform(feature_phn), nlen=7))
        array_labels += [labels[ii]] * len(feature_phn)

    return array_feature_replicated, array_labels, labels


def load_data_embedding_to_frame_level(list_feature, list_key, scaler):
    """load feature, scaling, replicate, reshape"""
    from audio_preprocessing import featureReshape

    array_feature_replicated, array_labels, labels = feature_replication(list_feature, list_key, scaler)

    # concatenate the feature of all phonemes into one array
    array_feature_replicated = np.concatenate(array_feature_replicated)
    array_feature_replicated = featureReshape(array_feature_replicated, nlen=7)

    return array_feature_replicated, np.array(array_labels)


def feature_replication_teacher_student(list_feature, list_key, scaler, data_str='_teacher'):
    from audio_preprocessing import _nbf_2D
    # flatten the feature and label list
    list_feature_flatten = []
    labels = []
    for ii in range(len(list_feature)):
        if list_key[ii] == u'sil' or list_key[ii] == u'?':
            pass
        else:
            list_feature_flatten += list_feature[ii]
            labels += [dic_pho_label_teacher_student[list_key[ii]+data_str]] * len(list_feature[ii])

    logger = logging.getLogger(__name__)
    logger.info('collect feature...')

    array_feature_replicated = []
    array_labels = []
    for ii in range(len(list_feature_flatten)):
        # pad the phn feature if < 7
        if list_feature_flatten[ii].shape[0] < 7:
            feature_phn = np.pad(list_feature_flatten[ii],
                                 ((0, 7 - list_feature_flatten[ii].shape[0]), (0, 0)),
                                 'edge')
        else:
            feature_phn = list_feature_flatten[ii]

        array_feature_replicated.append(_nbf_2D(scaler.transform(feature_phn), nlen=7))
        array_labels += [labels[ii]] * len(feature_phn)

    return array_feature_replicated, array_labels, labels


def load_data_embedding_to_frame_level_teacher_student(list_feature, list_key, scaler, data_str='_teacher'):
    """load feature, scaling, replicate, reshape"""
    from audio_preprocessing import featureReshape

    array_feature_replicated, array_labels, labels = feature_replication_teacher_student(list_feature, list_key, scaler, data_str)

    # concatenate the feature of all phonemes into one array
    array_feature_replicated = np.concatenate(array_feature_replicated)
    array_feature_replicated = featureReshape(array_feature_replicated, nlen=7)

    return array_feature_replicated, np.array(array_labels), labels


def cv5foldsIndices(list_feature_flatten, label_integer):
    """
    Have the split indices
    :param list_feature_flatten:
    :param label_integer:
    :return:
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    split_indices = []
    for train_index, val_index in skf.split(list_feature_flatten, label_integer):
        split_indices.append([train_index, val_index])
    return split_indices


def writeValLossCsv(file_path_log, ii_epoch, val_loss, train_loss=None):
    """
    write epoch number and validation loss to csv file
    :param file_path_log:
    :param ii_epoch:
    :param val_loss:
    :return:
    """
    append_write = append_or_write(file_path_log)
    with open(file_path_log, append_write) as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        if train_loss is not None:
            writer.writerow([ii_epoch, train_loss, val_loss])
        else:
            writer.writerow([ii_epoch, val_loss])