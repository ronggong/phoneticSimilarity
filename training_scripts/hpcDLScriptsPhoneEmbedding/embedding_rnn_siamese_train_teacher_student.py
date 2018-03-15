"""
lstm siamese train
"""

import sys, os

# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation import load_data_embedding_teacher_student
# from data_preparation import cv5foldsIndices
# from sklearn.model_selection import train_test_split
from models_siamese_tripletloss import train_embedding_siamese_batch_teacher_student
import pickle

import math
import numpy as np

def nPr(n,r):
    f = math.factorial
    return f(n) / f(n-r)

if __name__ == '__main__':

    batch_size = 1
    input_shape = (batch_size, None, 80)
    output_shape = int(sys.argv[2])
    # output_shape = 2
    patience = 15
    # margin = 0.15
    margin = float(sys.argv[1])

    filename_feature_teacher = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/feature_phn_embedding_train_teacher.pkl'
    filename_list_key_teacher = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/list_key_teacher.pkl'
    filename_feature_student = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/feature_phn_embedding_train_student.pkl'
    filename_list_key_student = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/list_key_student.pkl'
    filename_scaler = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/scaler_phn_embedding_train_teacher_student.pkl'
    filename_label_encoder = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/le_phn_embedding_teacher_student.pkl'
    filename_data_splits = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/data_splits_teacher_student.pkl'

    path_model = '/homedtic/rgong/phoneEmbeddingModelsTraining/out/'

    # path_dataset = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/phoneEmbedding'
    #
    # filename_feature_teacher = os.path.join(path_dataset, 'feature_phn_embedding_train_teacher.pkl')
    # filename_list_key_teacher = os.path.join(path_dataset, 'list_key_teacher.pkl')
    # filename_feature_student = os.path.join(path_dataset, 'feature_phn_embedding_train_student.pkl')
    # filename_list_key_student = os.path.join(path_dataset, 'list_key_student.pkl')
    # filename_scaler = os.path.join(path_dataset, 'scaler_phn_embedding_train_teacher_student.pkl')
    # filename_label_encoder = os.path.join(path_dataset, 'le_phn_embedding_teacher_student.pkl')
    # filename_data_splits = os.path.join(path_dataset, 'data_splits_teacher_student.pkl')
    #
    # path_model = '../../temp'

    list_feature_flatten, labels_integer, le, scaler = load_data_embedding_teacher_student(filename_feature_teacher,
                                                                                           filename_list_key_teacher,
                                                                                           filename_feature_student,
                                                                                           filename_list_key_student,
                                                                                           filename_scaler)

    if output_shape == 2:
        labels = le.inverse_transform(labels_integer)
        indices_teacher = [i for i, s in enumerate(labels) if 'teacher' in s]
        indices_student = [i for i, s in enumerate(labels) if 'student' in s]
        labels_integer[indices_teacher] = 0
        labels_integer[indices_student] = 1

    train_index, val_index = pickle.load(open(filename_data_splits, 'rb'))

    list_feature_fold_train = [scaler.transform(list_feature_flatten[ii]) for ii in train_index]
    labels_integer_fold_train = labels_integer[train_index]
    list_feature_fold_train = [np.expand_dims(feature, axis=0) for feature in list_feature_fold_train]

    list_feature_fold_val = [scaler.transform(list_feature_flatten[ii]) for ii in val_index]
    labels_integer_fold_val = labels_integer[val_index]
    list_feature_fold_val = [np.expand_dims(feature, axis=0) for feature in list_feature_fold_val]

    for ii in range(0, 5):
        if output_shape == 2:
            model_name = 'phone_embedding_RNN_triplet_teacher_student_margin_2_class'+sys.argv[1]
        else:
            model_name = 'phone_embedding_RNN_triplet_teacher_student_margin'+sys.argv[1]

        # model_name = 'phone_embedding_RNN_triplet_teacher_student_margin0.15'

        file_path_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
        file_path_log = os.path.join(path_model, 'log', model_name + '_' + str(ii) + '.csv')

        train_embedding_siamese_batch_teacher_student(list_feature_fold_train=list_feature_fold_train,
                                                      labels_fold_train=labels_integer_fold_train,
                                                      list_feature_fold_val=list_feature_fold_val,
                                                      labels_fold_val=labels_integer_fold_val,
                                                      batch_size=batch_size,
                                                      input_shape=input_shape,
                                                      output_shape=output_shape,
                                                      margin=margin,
                                                      file_path_model=file_path_model,
                                                      filename_log=file_path_log,
                                                      patience=patience,
                                                      reverse_anchor=False)

    # labels_train = labels_integer_fold_train
    # print(len(labels_train))
    #
    # import itertools
    #
    # idx_labels = np.arange(len(labels_train))
    #
    # num_same_paris = []
    # idx_same_pairs = []
    # for ii in range(29):
    #     idx_same_pairs_ii = np.asarray(list(itertools.permutations(idx_labels[labels_train == ii], r=2)))
    #     num_same_paris.append(len(idx_same_pairs_ii))
    #     idx_same_pairs.append(idx_same_pairs_ii)
    #
    # idx_same_pairs_reduced = []
    # for idx_same_pairs_ii in idx_same_pairs:
    #     idx_same_pairs_ii = idx_same_pairs_ii[np.random.choice(len(idx_same_pairs_ii), size=np.min(num_same_paris))]
    #     idx_same_pairs_reduced += list(idx_same_pairs_ii)
    #
    # print(len(idx_same_pairs_reduced))
    #
    # pair_percentage = 1
    # num_pairs_total = 0
    # for ii in range(29):
    #     num_ii = len(labels_train[labels_train==ii])
    #
    #     num_non_ii = len(labels_train) - num_ii
    #
    #     num_pairs_ii = nPr(n=num_ii, r=2)*pair_percentage
    #
    #     num_pairs_total += num_pairs_ii
    #
    # print(num_pairs_total)
