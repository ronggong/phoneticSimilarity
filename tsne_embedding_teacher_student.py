import matplotlib
matplotlib.use('Tkagg')

import os
# import csv
import numpy as np
import pickle
# import logging
# from src.distance_measures import gau_bh
# from sklearn.mixture import GaussianMixture
from training_scripts.data_preparation import load_data_embedding_teacher_student
from training_scripts.data_preparation import load_data_embedding_all
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from training_scripts.models_RNN import model_select
from src.parameters import config_select


def embedding_classifier_helper(configs,
                                MTL,
                                path_eval,
                                path_model,
                                le,
                                list_feature_flatten,
                                scaler,
                                label,
                                dense=False,
                                emb_all=False):

    embedding_dim = 32 if dense else 2
    for config in configs:
        input_shape = [1, None, 80]

        prefix = '_MTL' if MTL else '_2_class_teacher_student'
        model_name = config_select(config) + prefix if embedding_dim == 2 or embedding_dim == 32 else config_select(config)

        dense_str = "dense_32_" if dense else ""

        if emb_all and dense:
            emb_all_str = "_dense_all"
        elif emb_all:
            emb_all_str = "_all"
        else:
            emb_all_str = ""

        if le:
            # label encoder
            pickle.dump(le, open(os.path.join(path_eval, model_name + '_le.pkl'), 'wb'), protocol=2)

        for ii in range(1):
            filename_model = os.path.join(path_model, model_name + '_' + dense_str + str(ii) + '.h5')
            model = load_model(filepath=filename_model)
            weights = model.get_weights()

            x, input = model_select(config=config, input_shape=input_shape, conv=False, dropout=False)

            if MTL:
                pronun_out = Dense(27, activation='softmax', name='pronunciation')(x)
                profess_out = Dense(embedding_dim, activation='softmax', name='professionality')(x)
                model_1_batch = Model(inputs=input, outputs=[pronun_out, profess_out])
                model_1_batch.compile(optimizer='adam',
                                      loss='categorical_crossentropy',
                                      loss_weights=[0.5, 0.5])
            else:
                if dense:
                    outputs = Dense(embedding_dim)(x)
                else:
                    outputs = Dense(embedding_dim, activation='softmax')(x)
                model_1_batch = Model(inputs=input, outputs=outputs)

                model_1_batch.compile(optimizer='adam',
                                      loss='categorical_crossentropy',
                                      metrics=['accuracy'])
            model_1_batch.set_weights(weights=weights)

            embeddings_profess = np.zeros((len(list_feature_flatten), embedding_dim))

            if MTL:
                embeddings_pronun = np.zeros((len(list_feature_flatten), 27))

            for ii_emb in range(len(list_feature_flatten)):
                print('calculate', ii, 'run time', ii_emb, 'embedding', len(list_feature_flatten), 'total')

                x_batch = np.expand_dims(scaler.transform(list_feature_flatten[ii_emb]), axis=0)
                if MTL:
                    embeddings_pronun[ii_emb, :], embeddings_profess[ii_emb, :] = \
                        model_1_batch.predict_on_batch(x_batch)
                else:
                    embeddings_profess[ii_emb, :] = model_1_batch.predict_on_batch(x_batch)

            np.save(file=os.path.join(path_eval, model_name + '_embedding_professionality' + dense_str + emb_all_str + str(ii)),
                    arr=embeddings_profess)
            np.save(file=os.path.join(path_eval, model_name + '_embeddings_labels' + dense_str + emb_all_str), arr=label)

            if MTL:
                np.save(file=os.path.join(path_eval, model_name + '_embedding_pronunciation' + dense_str + emb_all_str + str(ii)),
                        arr=embeddings_pronun)


def embedding_classifier_tsne_all(filename_feature_teacher_train,
                                  filename_feature_teacher_val,
                                  filename_feature_teacher_test,
                                  filename_list_key_teacher,
                                  filename_feature_student_train,
                                  filename_feature_student_val,
                                  filename_feature_student_test,
                                  filename_list_key_student,
                                  filename_feature_student_extra_test,
                                  filename_list_key_extra_student,
                                  filename_scaler,
                                  MTL,
                                  dense):

    list_feature_flatten, list_key_flatten, scaler = load_data_embedding_all(filename_feature_teacher_train,
                                                                             filename_feature_teacher_val,
                                                                             filename_feature_teacher_test,
                                                                             filename_list_key_teacher,
                                                                             filename_feature_student_train,
                                                                             filename_feature_student_val,
                                                                             filename_feature_student_test,
                                                                             filename_list_key_student,
                                                                             filename_feature_student_extra_test,
                                                                             filename_list_key_extra_student,
                                                                             filename_scaler)

    path_model = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/models/phone_embedding_classifier'
    path_eval = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phone_embedding_classifier'

    configs = [[1, 0]]

    embedding_classifier_helper(configs,
                                MTL,
                                path_eval,
                                path_model,
                                le=False,
                                list_feature_flatten=list_feature_flatten,
                                scaler=scaler,
                                label=list_key_flatten,
                                emb_all=True,
                                dense=dense)


def embedding_classifier_tsne(filename_feature_teacher,
                              filename_list_key_teacher,
                              filename_feature_student,
                              filename_list_key_student,
                              filename_scaler,
                              MTL):
    """get the embedding of rnn classifier model"""

    list_feature_flatten_test, label_integer_test, le, scaler = \
        load_data_embedding_teacher_student(filename_feature_teacher=filename_feature_teacher,
                                            filename_list_key_teacher=filename_list_key_teacher,
                                            filename_feature_student=filename_feature_student,
                                            filename_list_key_student=filename_list_key_student,
                                            filename_scaler=filename_scaler)

    path_model = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/models/phone_embedding_classifier'
    path_eval = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phone_embedding_classifier'

    # configs = [[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]]
    configs = [[2, 0]]

    embedding_classifier_helper(configs,
                                MTL,
                                path_eval,
                                path_model,
                                le,
                                list_feature_flatten_test,
                                scaler,
                                label_integer_test)


if __name__ == '__main__':

    val_test = 'test'
    path_dataset = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/phoneEmbedding'

    # if val_test == 'val':
    #     filename_feature_teacher = os.path.join(path_dataset, 'feature_phn_embedding_val_teacher.pkl')
    #     filename_feature_student = os.path.join(path_dataset, 'feature_phn_embedding_val_student.pkl')
    #     filename_list_key_student = os.path.join(path_dataset, 'list_key_student.pkl')
    # elif val_test == 'test':
    #     filename_feature_teacher = os.path.join(path_dataset, 'feature_phn_embedding_test_teacher.pkl')
    #     filename_feature_student = os.path.join(path_dataset, 'feature_phn_embedding_test_extra_student.pkl')
    #     filename_list_key_student = os.path.join(path_dataset, 'list_key_extra_student.pkl')
    # else:
    #     raise ValueError('val test is not valid.')

    filename_feature_teacher_train = os.path.join(path_dataset, 'feature_phn_embedding_train_teacher.pkl')
    filename_feature_teacher_val = os.path.join(path_dataset, 'feature_phn_embedding_val_teacher.pkl')
    filename_feature_teacher_test = os.path.join(path_dataset, 'feature_phn_embedding_test_teacher.pkl')
    filename_list_key_teacher = os.path.join(path_dataset, 'list_key_teacher.pkl')

    filename_feature_student_train = os.path.join(path_dataset, 'feature_phn_embedding_train_student.pkl')
    filename_feature_student_val = os.path.join(path_dataset, 'feature_phn_embedding_val_student.pkl')
    filename_feature_student_test = os.path.join(path_dataset, 'feature_phn_embedding_test_student.pkl')
    filename_list_key_student = os.path.join(path_dataset, 'list_key_student.pkl')

    filename_feature_student_extra_test = os.path.join(path_dataset, 'feature_phn_embedding_test_extra_student.pkl')
    filename_list_key_extra_student = os.path.join(path_dataset, 'list_key_extra_student.pkl')

    filename_scaler = os.path.join(path_dataset, 'scaler_phn_embedding_train_teacher_student.pkl')

    embedding_classifier_tsne_all(filename_feature_teacher_train,
                                  filename_feature_teacher_val,
                                  filename_feature_teacher_test,
                                  filename_list_key_teacher,
                                  filename_feature_student_train,
                                  filename_feature_student_val,
                                  filename_feature_student_test,
                                  filename_list_key_student,
                                  filename_feature_student_extra_test,
                                  filename_list_key_extra_student,
                                  filename_scaler,
                                  MTL=False,
                                  dense=True)

    # embedding_classifier_tsne(filename_feature_teacher_train,
    #                           filename_feature_teacher_val,
    #                           filename_feature_teacher_test,
    #                           filename_list_key_teacher,
    #                           filename_feature_student_train,
    #                           filename_feature_student_val,
    #                           filename_feature_student_test,
    #                           filename_list_key_student,
    #                           filename_feature_student_extra_test,
    #                           filename_list_key_extra_student,
    #                           filename_scaler,
    #                           MTL=False)

    # embedding_siamese_tsne(filename_feature, filename_list_key, filename_scaler)
    # embedding_frame_tsne(filename_feature, filename_list_key, filename_scaler)

    # correlationDistanceMat()

