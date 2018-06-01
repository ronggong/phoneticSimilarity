"""
train the lstm classifier embedding
"""
import sys
import os
import pickle

# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation import load_data_embedding_teacher_student
from keras.utils import to_categorical
from models_RNN import train_embedding_RNN_batch_MTL
from parameters import config_select

# from data_preparation import cv5foldsIndices
# from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    batch_size = 64
    input_shape = (batch_size, None, 80)
    patience = 15
    output_shape = [27, 2]
    attention = False
    dense = False
    conv = True
    if attention:
        attention_dense_str = "attention_"
    elif dense:
        attention_dense_str = "dense_"
    elif conv:
        attention_dense_str = "conv_"
    else:
        attention_dense_str = ""

    path_dataset = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/'

    filename_feature_teacher = os.path.join(path_dataset, 'feature_phn_embedding_train_teacher.pkl')
    filename_list_key_teacher = os.path.join(path_dataset, 'list_key_teacher.pkl')
    filename_feature_student = os.path.join(path_dataset, 'feature_phn_embedding_train_student.pkl')
    filename_list_key_student = os.path.join(path_dataset, 'list_key_student.pkl')
    filename_scaler_teacher_student = os.path.join(path_dataset, 'scaler_phn_embedding_train_teacher_student.pkl')

    filename_label_encoder = os.path.join(path_dataset, 'le_phn_embedding_teacher_student.pkl')
    filename_data_splits = os.path.join(path_dataset, 'data_splits_teacher_student.pkl')

    path_model = '/homedtic/rgong/phoneEmbeddingModelsTraining/out/'

    # path_dataset = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/phoneEmbedding'
    #
    # filename_feature_teacher = os.path.join(path_dataset, 'feature_phn_embedding_train_teacher.pkl')
    # filename_list_key_teacher = os.path.join(path_dataset, 'list_key_teacher.pkl')
    # filename_feature_student = os.path.join(path_dataset, 'feature_phn_embedding_train_student.pkl')
    # filename_list_key_student = os.path.join(path_dataset, 'list_key_student.pkl')
    #
    # filename_scaler_teacher_student = os.path.join(path_dataset, 'scaler_phn_embedding_train_teacher_student.pkl')
    #
    # filename_label_encoder = os.path.join(path_dataset, 'le_phn_embedding_teacher_student.pkl')
    # filename_data_splits = os.path.join(path_dataset, 'data_splits_teacher_student.pkl')
    #
    # path_model = '../../temp'

    list_feature_flatten, labels_integer, le, scaler = \
                load_data_embedding_teacher_student(filename_feature_teacher=filename_feature_teacher,
                                                    filename_list_key_teacher=filename_list_key_teacher,
                                                    filename_feature_student=filename_feature_student,
                                                    filename_list_key_student=filename_list_key_student,
                                                    filename_scaler=filename_scaler_teacher_student)

    labels_integer_phn = labels_integer.copy()
    labels_integer_pro = labels_integer.copy()
    # combine teacher and student label to the same one
    labels = le.inverse_transform(labels_integer)
    phn_set = list(set([l.split('_')[0] for l in labels]))
    for ii in range(len(phn_set)):
        indices_phn = [i for i, s in enumerate(labels) if phn_set[ii] == s.split('_')[0]]
        labels_integer_phn[indices_phn] = ii

    # professionality labels
    indices_teacher = [i for i, s in enumerate(labels) if 'teacher' in s]
    indices_student = [i for i, s in enumerate(labels) if 'student' in s]
    labels_integer_pro[indices_teacher] = 0
    labels_integer_pro[indices_student] = 1

    # split folds
    # folds5_split_indices = cv5foldsIndices(list_feature_flatten=list_feature_flatten, label_integer=labels_integer)

    # index_feature = range(len(list_feature_flatten))
    # train_index, val_index, _, _ = train_test_split(index_feature, labels_integer, test_size=0.1, stratify=labels_integer)
    #
    # pickle.dump(le ,open(filename_label_encoder, 'wb'), protocol=2)
    # pickle.dump([train_index, val_index], open(filename_data_splits, 'wb'), protocol=2)

    train_index, val_index = pickle.load(open(filename_data_splits, 'rb'))
    #
    # for train_index, val_index in folds5_split_indices:
    if attention or dense or conv:
        configs = [[2, 0]]
    else:
        configs = [[1, 1], [1, 0], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]]

    for config in configs:

        model_name = config_select(config=config)

        for ii in range(5):
            file_path_model = os.path.join(path_model, model_name + '_MTL' + '_' + attention_dense_str + str(ii) + '.h5')
            file_path_log = os.path.join(path_model, 'log', model_name + '_MTL' + '_' + attention_dense_str + str(ii) + '.csv')

            list_feature_fold_train = [scaler.transform(list_feature_flatten[ii]) for ii in train_index]
            labels_integer_phn_fold_train = labels_integer_phn[train_index]
            labels_integer_pro_fold_train = labels_integer_pro[train_index]
            labels_phn_fold_train = to_categorical(labels_integer_phn_fold_train)
            labels_pro_fold_train = to_categorical(labels_integer_pro_fold_train)
            labels_fold_train = [phn_pro for phn_pro in zip(labels_phn_fold_train, labels_pro_fold_train)]

            list_feature_fold_val = [scaler.transform(list_feature_flatten[ii]) for ii in val_index]
            labels_integer_phn_fold_val = labels_integer_phn[val_index]
            labels_integer_pro_fold_val = labels_integer_pro[val_index]
            labels_phn_fold_val = to_categorical(labels_integer_phn_fold_val)
            labels_pro_fold_val = to_categorical(labels_integer_pro_fold_val)
            labels_fold_val = [phn_pro for phn_pro in zip(labels_phn_fold_val, labels_pro_fold_val)]

            # print(len(list_feature_fold_train), len(labels_fold_train))
            # print(len(list_feature_fold_val), len(labels_fold_val))

            train_embedding_RNN_batch_MTL(list_feature_fold_train=list_feature_fold_train,
                                          labels_fold_train=labels_fold_train,
                                          list_feature_fold_val=list_feature_fold_val,
                                          labels_fold_val=labels_fold_val,
                                          batch_size=batch_size,
                                          input_shape=input_shape,
                                          output_shape=output_shape,
                                          file_path_model=file_path_model,
                                          filename_log=file_path_log,
                                          patience=patience,
                                          config=config,
                                          attention=attention,
                                          dense=dense,
                                          conv=conv)

