"""
lstm siamese train
"""

import sys, os

# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation import load_data_embedding
# from data_preparation import cv5foldsIndices
# from sklearn.model_selection import train_test_split
from models_siamese_tripletloss import train_embedding_siamese_Ndiff_train_fit_generator_val_routine
import pickle

# import math
import numpy as np


if __name__ == '__main__':

    batch_size=128
    input_shape = (batch_size, None, 80)
    output_shape = 29
    patience=15
    # margin=sys.argv[1]
    margin = str(0.15)

    filename_feature = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/feature_phn_embedding_train.pkl'
    filename_list_key = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/list_key.pkl'
    filename_scaler = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/scaler_phn_embedding.pkl'
    filename_label_encoder = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/le_phn_embedding.pkl'
    filename_data_splits = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/data_splits.pkl'

    path_model = '/homedtic/rgong/phoneEmbeddingModelsTraining/out/'

    # path_dataset = '/Users/ronggong/Documents_using/MTG document/dataset/phoneEmbedding'
    #
    # filename_feature = os.path.join(path_dataset, 'feature_phn_embedding_train.pkl')
    # filename_list_key = os.path.join(path_dataset, 'list_key.pkl')
    # filename_scaler = os.path.join(path_dataset, 'scaler_phn_embedding.pkl')
    # filename_label_encoder = os.path.join(path_dataset, 'le_phn_embedding.pkl')
    # filename_data_splits = os.path.join(path_dataset, 'data_splits.pkl')
    #
    # # path_model = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/models/phone_embedding_classifier/'
    # path_model = '../../temp'

    list_feature_flatten, labels_integer, le, scaler = load_data_embedding(filename_feature=filename_feature,
                                                                           filename_list_key=filename_list_key,
                                                                           filename_scaler=filename_scaler)
    train_index, val_index = pickle.load(open(filename_data_splits, 'rb'))

    list_feature_fold_train = [scaler.transform(list_feature_flatten[ii]) for ii in train_index]
    labels_integer_fold_train = labels_integer[train_index]
    list_feature_fold_train = [np.expand_dims(feature, axis=0) for feature in list_feature_fold_train]

    list_feature_fold_val = [scaler.transform(list_feature_flatten[ii]) for ii in val_index]
    labels_integer_fold_val = labels_integer[val_index]
    list_feature_fold_val = [np.expand_dims(feature, axis=0) for feature in list_feature_fold_val]

    for ii in range(0, 5):
        model_name = 'phone_embedding_RNN_triplet_Ndiff5_margin_cpu'+margin

        file_path_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
        file_path_log = os.path.join(path_model, 'log', model_name + '_' + str(ii) + '.csv')

        train_embedding_siamese_Ndiff_train_fit_generator_val_routine(list_feature_fold_train=list_feature_fold_train,
                                                                      labels_fold_train=labels_integer_fold_train,
                                                                      list_feature_fold_val=list_feature_fold_val,
                                                                      labels_fold_val=labels_integer_fold_val,
                                                                      batch_size=batch_size,
                                                                      input_shape=input_shape,
                                                                      output_shape=output_shape,
                                                                      N_diff=5,
                                                                      margin=float(margin),
                                                                      file_path_model=file_path_model,
                                                                      file_path_log=file_path_log,
                                                                      patience=patience,
                                                                      verbose=2,
                                                                      reverse_anchor=False)