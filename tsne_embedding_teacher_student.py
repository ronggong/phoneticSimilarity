import matplotlib
matplotlib.use('Tkagg')

import os
# import csv
import numpy as np
import pickle
# import logging
# from src.distance_measures import gau_bh
# from sklearn.mixture import GaussianMixture
from data_preparation import load_data_embedding_teacher_student
from keras.models import load_model
# from keras.models import Model
from models_RNN import model_select
from parameters import config_select



def embedding_classifier_tsne(filename_feature_teacher,
                              filename_list_key_teacher,
                              filename_feature_student,
                              filename_list_key_student,
                              filename_scaler):
    """get the embedding of rnn classifier model"""

    list_feature_flatten_test, label_integer_test, le, scaler = \
        load_data_embedding_teacher_student(filename_feature_teacher=filename_feature_teacher,
                                            filename_list_key_teacher=filename_list_key_teacher,
                                            filename_feature_student=filename_feature_student,
                                            filename_list_key_student=filename_list_key_student,
                                            filename_scaler=filename_scaler)

    path_model = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/models/phone_embedding_classifier'
    path_eval = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phone_embedding_classifier'

    # configs = [[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]]
    configs = [[1, 1]]

    for config in configs:
        model_name = config_select(config)
        pickle.dump(le, open(os.path.join(path_eval, model_name + '_teacher_student' + '_le.pkl'), 'wb'), protocol=2)

        embedding_dim = 54
        input_shape = [1, None, 80]

        for ii in range(1):
            filename_model = os.path.join(path_model, model_name + '_teacher_student' + '_' + str(ii) + '.h5')
            model = load_model(filepath=filename_model)
            weights = model.get_weights()

            model_1_batch = model_select(config=config, input_shape=input_shape, output_shape=embedding_dim)
            model_1_batch.compile(optimizer='adam',
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])
            model_1_batch.set_weights(weights=weights)

            embeddings = np.zeros((len(list_feature_flatten_test), embedding_dim))
            for ii_emb in range(len(list_feature_flatten_test)):
                print('calculate', ii, 'run time', ii_emb, 'embedding', len(list_feature_flatten_test), 'total')

                x_batch = np.expand_dims(scaler.transform(list_feature_flatten_test[ii_emb]), axis=0)
                embeddings[ii_emb, :] = model_1_batch.predict_on_batch(x_batch)

            np.save(file=os.path.join(path_eval, model_name + '_teacher_student' + '_embedding_' + str(ii)), arr=embeddings)
            np.save(file=os.path.join(path_eval, model_name + '_teacher_student' + '_labels'), arr=label_integer_test)


if __name__ == '__main__':

    filename_feature_teacher = '/home/gong/Documents/MTG/dataset/phoneEmbedding/feature_phn_embedding_test_teacher.pkl'
    filename_list_key_teacher = '/home/gong/Documents/MTG/dataset/phoneEmbedding/list_key_teacher.pkl'
    filename_feature_student = '/home/gong/Documents/MTG/dataset/phoneEmbedding/feature_phn_embedding_test_student.pkl'
    filename_list_key_student = '/home/gong/Documents/MTG/dataset/phoneEmbedding/list_key_student.pkl'
    filename_scaler = '/home/gong/Documents/MTG/dataset/phoneEmbedding/scaler_phn_embedding_train_teacher_student.pkl'

    embedding_classifier_tsne(filename_feature_teacher,
                              filename_list_key_teacher,
                              filename_feature_student,
                              filename_list_key_student,
                              filename_scaler)

    # embedding_siamese_tsne(filename_feature, filename_list_key, filename_scaler)
    # embedding_frame_tsne(filename_feature, filename_list_key, filename_scaler)

    # correlationDistanceMat()

