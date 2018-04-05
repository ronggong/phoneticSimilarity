"""
script to train the frame level embedding
"""

import sys
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation import load_data_embedding_to_frame_level_teacher_student
from models import train_model_validation
import numpy as np
import pickle
import h5py
import atexit


def exit_handler(filenames):
    """delete files when exits"""
    for fn in filenames:
        if os.path.isfile(fn):
            os.remove(fn)


def main():

    tmp_folder = '/tmp/phoneEmbeddingModelsTraining'
    if not os.path.isdir(tmp_folder):
        os.mkdir(tmp_folder)

    path_dataset = '/homedtic/rgong/phoneEmbeddingModelsTraining/dataset/'

    filename_feature_teacher = os.path.join(path_dataset, 'feature_phn_embedding_train_teacher.pkl')
    filename_list_key_teacher = os.path.join(path_dataset, 'list_key_teacher.pkl')
    filename_feature_student = os.path.join(path_dataset, 'feature_phn_embedding_train_student.pkl')
    filename_list_key_student = os.path.join(path_dataset, 'list_key_student.pkl')
    filename_scaler = os.path.join(path_dataset, 'scaler_phn_embedding_train_teacher_student.pkl')

    filename_train_validation_set = os.path.join(tmp_folder, 'feature_frame.h5')
    filename_labels_train_validation_set = os.path.join(tmp_folder, 'labels.pkl')

    path_model = '/homedtic/rgong/phoneEmbeddingModelsTraining/out/'

    # path_dataset = '/Users/ronggong/Documents_using/MTG document/dataset/phoneEmbedding'
    #
    # filename_feature_teacher = os.path.join(path_dataset, 'feature_phn_embedding_train_teacher.pkl')
    # filename_list_key_teacher = os.path.join(path_dataset, 'list_key_teacher.pkl')
    # filename_feature_student = os.path.join(path_dataset, 'feature_phn_embedding_train_student.pkl')
    # filename_list_key_student = os.path.join(path_dataset, 'list_key_student.pkl')
    #
    # filename_scaler = os.path.join(path_dataset, 'scaler_phn_embedding_train_teacher_student.pkl')
    #
    # filename_train_validation_set = '../../temp/feature_frame.h5'
    # filename_labels_train_validation_set = '../../temp/labels.pkl'
    #
    # path_model = '../../temp'

    input_dim = (80, 15)
    output_shape = 2  # 54

    # feature, label, scaler loading
    list_feature_teacher = pickle.load(open(filename_feature_teacher, 'rb'))
    list_key_teacher = pickle.load(open(filename_list_key_teacher, 'rb'))
    list_feature_student = pickle.load(open(filename_feature_student, 'rb'))
    list_key_student = pickle.load(open(filename_list_key_student, 'rb'))
    scaler = pickle.load(open(filename_scaler, 'rb'))

    array_feature_replicated_teacher, array_labels_teacher, labels_teacher = \
        load_data_embedding_to_frame_level_teacher_student(list_feature=list_feature_teacher,
                                                           list_key=list_key_teacher,
                                                           scaler=scaler,
                                                           data_str='_teacher')

    array_feature_replicated_student, array_labels_student, labels_student = \
        load_data_embedding_to_frame_level_teacher_student(list_feature=list_feature_student,
                                                           list_key=list_key_student,
                                                           scaler=scaler,
                                                           data_str='_student')

    array_feature_replicated = \
        np.concatenate((array_feature_replicated_teacher, array_feature_replicated_student), axis=0)

    array_labels = np.concatenate((array_labels_teacher, array_labels_student))

    # 2 class case
    if output_shape == 2:
        array_labels[array_labels <= 26] = 0
        array_labels[array_labels > 26] = 1
        model_name = 'wide_frame_level_emb_teacher_student_2_class'
    else:
        model_name = 'wide_frame_level_emb_teacher_student'

    # write feature and label to files
    h5f = h5py.File(filename_train_validation_set, 'w')
    h5f.create_dataset('feature_all', data=array_feature_replicated)
    h5f.close()

    pickle.dump(array_labels, open(filename_labels_train_validation_set, 'wb'))

    for ii in range(5):
        file_path_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
        file_path_log = os.path.join(path_model, 'log', model_name + '_' + str(ii) + '.csv')

        train_model_validation(filename_train_validation_set=filename_train_validation_set,
                               filename_labels_train_validation_set=filename_labels_train_validation_set,
                               filter_density=4,
                               dropout=0.32,
                               input_shape=input_dim,
                               output_shape=output_shape,
                               file_path_model=file_path_model,
                               filename_log=file_path_log,
                               channel=1)

    # clean the feature file
    atexit.register(exit_handler, filenames=[filename_train_validation_set, filename_labels_train_validation_set])


if __name__ == '__main__':
    main()