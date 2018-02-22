"""
train the lstm classifier embedding
"""
import sys, os

# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation import load_data_embedding
# from data_preparation import cv5foldsIndices
# from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from models_RNN import train_embedding_RNN_batch
from parameters import config_select
import pickle

if __name__ == '__main__':

    batch_size=64
    input_shape = (batch_size, None, 80)
    patience=15

    # filename_feature = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/feature_phn_embedding_train.pkl'
    # filename_list_key = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/list_key.pkl'
    # filename_scaler = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/scaler_phn_embedding.pkl'
    # filename_label_encoder = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/le_phn_embedding.pkl'
    # filename_data_splits = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/data_splits.pkl'
    #
    # path_model = '/homedtic/rgong/phoneEmbeddingModelsTraining/out/'

    filename_feature = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/feature_phn_embedding_train.pkl'
    filename_list_key = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/list_key.pkl'
    filename_scaler = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/scaler_phn_embedding.pkl'
    filename_label_encoder = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/le_phn_embedding.pkl'
    filename_data_splits = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/data_splits.pkl'

    path_model = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/models/phone_embedding_classifier/'
    # path_model = '../temp'

    list_feature_flatten, labels_integer, le, scaler = load_data_embedding(filename_feature=filename_feature,
                                                                             filename_list_key=filename_list_key,
                                                                             filename_scaler=filename_scaler)

    # # split folds
    # folds5_split_indices = cv5foldsIndices(list_feature_flatten=list_feature_flatten, label_integer=labels_integer)

    # index_feature = range(len(list_feature_flatten))
    # train_index, val_index, _, _ = train_test_split(index_feature, labels_integer, test_size=0.1, stratify=labels_integer)
    #
    # pickle.dump(le ,open(filename_label_encoder, 'wb'), protocol=2)
    # pickle.dump([train_index, val_index], open(filename_data_splits, 'wb'), protocol=2)

    train_index, val_index = pickle.load(open(filename_data_splits, 'rb'))
    #
    # for train_index, val_index in folds5_split_indices:

    configs = [[3, 1], [3, 2], [3, 3]]


    for config in configs:

        model_name = config_select(config=config)

        for ii in range(5):
            file_path_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
            file_path_log = os.path.join(path_model, 'log', model_name + '_' + str(ii) + '.csv')


            list_feature_fold_train = [scaler.transform(list_feature_flatten[ii]) for ii in train_index]
            labels_integer_fold_train = labels_integer[train_index]
            labels_fold_train = to_categorical(labels_integer_fold_train)

            list_feature_fold_val = [scaler.transform(list_feature_flatten[ii]) for ii in val_index]
            labels_integer_fold_val = labels_integer[val_index]
            labels_fold_val = to_categorical(labels_integer_fold_val)

            print(len(list_feature_fold_train), len(labels_fold_train))
            print(len(list_feature_fold_val), len(labels_fold_val))

            train_embedding_RNN_batch(list_feature_fold_train=list_feature_fold_train,
                                      labels_fold_train=labels_fold_train,
                                      list_feature_fold_val=list_feature_fold_val,
                                      labels_fold_val=labels_fold_val,
                                      batch_size=batch_size,
                                      input_shape=input_shape,
                                      file_path_model=file_path_model,
                                      filename_log=file_path_log,
                                      patience=patience,
                                      config=config)



        # train_embedding_RNN(X_train=X_train,
        #                     X_val=X_val,
        #                     y_train=y_train,
        #                     y_val=y_val,
        #                     scaler=scaler,
        #                     input_shape=input_shape,
        #                     file_path_model=file_path_model,
        #                     file_path_log=file_path_log)