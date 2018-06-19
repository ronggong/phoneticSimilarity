import os
import pickle
import numpy as np
from scipy.spatial.distance import cosine
from src.parameters import *
from keras.models import load_model
from keras.layers import Dense
from keras.models import Model
from training_scripts.models_RNN import model_select
from training_scripts.models_RNN import model_select_attention
from training_scripts.attention import Attention


def load_log_mel(filename):
    """
    Load log mel pickle, cut them into phoneme chucks
    :param filename:
    :return:
    """
    log_mel, list_phn = pickle.load(open(filename, "rb"))
    list_feature = []
    list_phn_name = []
    for phn in list_phn:
        if len(phn[2]) > 0:
            start_frame = int(round((phn[0] - list_phn[0][0]) * fs / float(hopsize)))
            end_frame = int(round((phn[1] - list_phn[0][0]) * fs / float(hopsize)))
            list_feature.append(log_mel[start_frame: end_frame])
            list_phn_name.append(phn[2])
    return list_feature, list_phn_name


def calculate_pronunciation_score(list_feature_teacher, list_feature_student):
    """
    Calculate pronunciation similarity
    :param list_feature_teacher:
    :param list_feature_student:
    :return:
    """
    path_model = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/models/phone_embedding_classifier'
    path_dataset = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/phoneEmbedding'
    conv = True
    dropout = True
    config = [2, 0]
    input_shape = [1, None, 80]
    embedding_dim = 27
    prefix = '_27_class'
    attention_dense_str = 'attention_conv_dropout_'
    model_name = config_select(config)
    filename_model = os.path.join(path_model, model_name + prefix + '_' + attention_dense_str + str(0) + '.h5')
    filename_scaler = os.path.join(path_dataset, 'scaler_phn_embedding_train_teacher_student.pkl')

    # loading models
    model = load_model(filepath=filename_model, custom_objects={'Attention': Attention(return_attention=True)})
    weights = model.get_weights()
    x, input, _ = model_select_attention(config=config, input_shape=input_shape, conv=conv, dropout=dropout)
    outputs = Dense(embedding_dim, activation='softmax')(x)
    model_1_batch = Model(inputs=input, outputs=outputs)
    model_1_batch.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    model_1_batch.set_weights(weights=weights)
    scaler = pickle.load(open(filename_scaler, 'rb'))

    # calculate the similarity
    list_simi = []
    for ii in range(len(list_feature_teacher)):
        x_batch_teacher = np.expand_dims(scaler.transform(list_feature_teacher[ii]), axis=0)
        x_batch_student = np.expand_dims(scaler.transform(list_feature_student[ii]), axis=0)
        embedding_teacher = model_1_batch.predict_on_batch(x_batch_teacher)
        embedding_student = model_1_batch.predict_on_batch(x_batch_student)
        dist = (2.0 - cosine(embedding_teacher, embedding_student)) / 2.0
        list_simi.append(dist)
    return list_simi


def calculate_overall_quality_score(list_feature_teacher, list_feature_student):
    """
    Calculate overall quality similarity
    :param list_feature_teacher:
    :param list_feature_student:
    :return:
    """
    path_model = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/models/phone_embedding_classifier'
    path_dataset = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/phoneEmbedding'
    config = [1, 0]
    input_shape = [1, None, 80]
    embedding_dim = 32
    prefix = '_2_class_teacher_student'
    attention_dense_str = "dense_32_"
    model_name = config_select(config) + prefix
    filename_model = os.path.join(path_model, model_name + '_' + attention_dense_str + str(0) + '.h5')
    filename_scaler = os.path.join(path_dataset, 'scaler_phn_embedding_train_teacher_student.pkl')

    model = load_model(filepath=filename_model, custom_objects={'Attention': Attention(return_attention=True)})
    weights = model.get_weights()
    x, input = model_select(config=config, input_shape=input_shape, conv=False, dropout=False)
    outputs = Dense(embedding_dim)(x)
    model_1_batch = Model(inputs=input, outputs=outputs)
    model_1_batch.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    model_1_batch.set_weights(weights=weights)
    scaler = pickle.load(open(filename_scaler, 'rb'))

    list_simi = []
    for ii in range(len(list_feature_teacher)):
        x_batch_teacher = np.expand_dims(scaler.transform(list_feature_teacher[ii]), axis=0)
        x_batch_student = np.expand_dims(scaler.transform(list_feature_student[ii]), axis=0)
        embedding_teacher = model_1_batch.predict_on_batch(x_batch_teacher)
        embedding_student = model_1_batch.predict_on_batch(x_batch_student)
        dist = (2.0 - cosine(embedding_teacher, embedding_student)) / 2.0
        list_simi.append(dist)
    return list_simi


if __name__ == "__main__":
    plotting_data_folder = "../data/plotting_data/"

    # yang yu huan
    list_feature_teacher_yang, list_phn_teacher_yang = \
        load_log_mel(filename=os.path.join(plotting_data_folder, "teacher_yang_yu_huan.pkl"))
    list_feature_student_yang, list_phn_student_yang = \
        load_log_mel(filename=os.path.join(plotting_data_folder, "student_yang_yu_huan.pkl"))

    extra_teacher_phns = [2, 24, 25, 31]
    list_feature_teacher_yang = [list_feature_teacher_yang[ii] for ii in range(len(list_feature_teacher_yang)) if ii not in extra_teacher_phns]

    list_simi_pronunciation_yang = calculate_pronunciation_score(list_feature_teacher=list_feature_teacher_yang,
                                                                 list_feature_student=list_feature_student_yang)

    list_simi_overall_quality_yang = calculate_overall_quality_score(list_feature_teacher=list_feature_teacher_yang,
                                                                     list_feature_student=list_feature_student_yang)
    pickle.dump(list_simi_pronunciation_yang,
                open(os.path.join(plotting_data_folder, "simi_pronunciation_yang_yu_huan.pkl"), "wb"))
    pickle.dump(list_simi_overall_quality_yang,
                open(os.path.join(plotting_data_folder, "simi_overall_quality_yang_yu_huan.pkl"), "wb"))
    pickle.dump(extra_teacher_phns,
                open(os.path.join(plotting_data_folder, "extra_teacher_phns_yang_yu_huan.pkl"), "wb"))

    # meng ting de
    list_feature_teacher_meng, list_phn_teacher_meng = \
        load_log_mel(filename=os.path.join(plotting_data_folder, "teacher_meng_ting_de.pkl"))
    list_feature_student_meng, list_phn_student_meng = \
        load_log_mel(filename=os.path.join(plotting_data_folder, "student_meng_ting_de.pkl"))

    extra_teacher_phns = [2, 5, 10, 16, 25, 28]
    list_feature_teacher_meng = [list_feature_teacher_meng[ii] for ii in range(len(list_feature_teacher_meng)) if
                                 ii not in extra_teacher_phns]

    list_simi_pronunciation_meng = calculate_pronunciation_score(list_feature_teacher=list_feature_teacher_meng,
                                                                 list_feature_student=list_feature_student_meng)

    list_simi_overall_quality_meng = calculate_overall_quality_score(list_feature_teacher=list_feature_teacher_meng,
                                                                     list_feature_student=list_feature_student_meng)

    pickle.dump(list_simi_pronunciation_meng,
                open(os.path.join(plotting_data_folder, "simi_pronunciation_meng_ting_de.pkl"), "wb"))
    pickle.dump(list_simi_overall_quality_meng,
                open(os.path.join(plotting_data_folder, "simi_overall_quality_meng_ting_de.pkl"), "wb"))
    pickle.dump(extra_teacher_phns,
                open(os.path.join(plotting_data_folder, "extra_teacher_phns_meng_ting_de.pkl"), "wb"))

