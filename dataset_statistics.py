"""
Code for calculating the training, validation and testing dataset statistics
"""
import os
import pickle
from data_preparation import featureFlatten


def load_data_teacher_student(filename_feature_teacher,
                              filename_list_key_teacher,
                              filename_feature_student,
                              filename_list_key_student):
    """
    load data for calculate statistics
    """
    list_feature_teacher = pickle.load(open(filename_feature_teacher, 'rb'))
    list_key_teacher = pickle.load(open(filename_list_key_teacher, 'rb'))
    list_feature_student = pickle.load(open(filename_feature_student, 'rb'))
    list_key_student = pickle.load(open(filename_list_key_student, 'rb'))

    # flatten the feature and label list
    list_feature_flatten_teacher, list_key_flatten_teacher = \
        featureFlatten(list_feature_teacher, list_key_teacher, '_teacher')
    list_feature_flatten_student, list_key_flatten_student = \
        featureFlatten(list_feature_student, list_key_student, '_student')

    # list_feature_flatten = list_feature_flatten_teacher + list_feature_flatten_student
    list_key_flatten = list_key_flatten_teacher + list_key_flatten_student
    return list_key_flatten


if __name__ == '__main__':
    path_dataset = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/phoneEmbedding'

    val_test = 'test'

    phonemes = ['S', 'EnEn', 'O', 'nvc', 'N', 'j', 'in', 'y', '@n', 'i', 'MM', 'oU^', 'SN', 'aI^', 'an', 'AU^', 'rr', 'ANAN', '@', 'a', 'vc', 'iNiN', 'eI^', 'UN', 'u', 'E', 'ONE']

    if val_test == 'train':
        filename_feature_teacher = os.path.join(path_dataset, 'feature_phn_embedding_train_teacher.pkl')
        filename_feature_student = os.path.join(path_dataset, 'feature_phn_embedding_train_student.pkl')
        filename_list_key_student = os.path.join(path_dataset, 'list_key_student.pkl')
    elif val_test == 'val':
        filename_feature_teacher = os.path.join(path_dataset, 'feature_phn_embedding_val_teacher.pkl')
        filename_feature_student = os.path.join(path_dataset, 'feature_phn_embedding_val_student.pkl')
        filename_list_key_student = os.path.join(path_dataset, 'list_key_student.pkl')
    elif val_test == 'test':
        filename_feature_teacher = os.path.join(path_dataset, 'feature_phn_embedding_test_teacher.pkl')
        filename_feature_student = os.path.join(path_dataset, 'feature_phn_embedding_test_extra_student.pkl')
        filename_list_key_student = os.path.join(path_dataset, 'list_key_extra_student.pkl')
    else:
        raise ValueError('val test is not valid.')

    filename_list_key_teacher = os.path.join(path_dataset, 'list_key_teacher.pkl')

    list_key = load_data_teacher_student(filename_feature_teacher=filename_feature_teacher,
                                         filename_list_key_teacher=filename_list_key_teacher,
                                         filename_feature_student=filename_feature_student,
                                         filename_list_key_student=filename_list_key_student)

    # phonemes = list(set([key.split('_')[0] for key in list(set(list_key))]))

    for p in phonemes:
        print(p, list_key.count(p+'_teacher'), list_key.count(p+'_student'))