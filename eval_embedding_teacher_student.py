"""this script evaluate the embedding trained by separating the phonemes into 54 classes: 27 phone class * 2 pro/amatuer,
embedding_classifier_ap evaluate the classifier model,
embedding_frame_ap evaluate the frame-based model,
embedding_siamese_ap evaluate the siamese model"""

import csv
import os
import pickle
import logging
import numpy as np
from eval_embedding import ground_truth_matrix
from eval_embedding import eval_embeddings
from data_preparation import load_data_embedding_teacher_student
from data_preparation import feature_replication_teacher_student
from parameters import config_select
from models_RNN import model_select
from keras.models import load_model
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from src.audio_preprocessing import featureReshape

from training_scripts.models_siamese_tripletloss import embedding_2_lstm_1_dense_base
from training_scripts.models_siamese_tripletloss import embedding_base_model


def embedding_classifier_ap(filename_feature_teacher,
                            filename_list_key_teacher,
                            filename_feature_student,
                            filename_list_key_student,
                            filename_scaler,
                            config,
                            val_test):
    """calculate average precision of classifier embedding"""

    list_feature_flatten_val, label_integer_val, le, scaler = \
        load_data_embedding_teacher_student(filename_feature_teacher=filename_feature_teacher,
                                            filename_list_key_teacher=filename_list_key_teacher,
                                            filename_feature_student=filename_feature_student,
                                            filename_list_key_student=filename_list_key_student,
                                            filename_scaler=filename_scaler)

    path_model = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/models/phone_embedding_classifier'
    path_eval = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phone_embedding_classifier'

    # configs = [[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]]
    # configs = [[1, 1]]

    # for config in configs:
    model_name = config_select(config)

    list_ap = []
    embedding_dim = 54

    for ii in range(5):
        filename_model = os.path.join(path_model, model_name + '_teacher_student' + '_' + str(ii) + '.h5')
        model = load_model(filepath=filename_model)
        weights = model.get_weights()

        input_shape = [1, None, 80]
        model_1_batch = model_select(config=config, input_shape=input_shape, output_shape=embedding_dim)
        model_1_batch.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
        model_1_batch.set_weights(weights=weights)

        embeddings = np.zeros((len(list_feature_flatten_val), embedding_dim))
        for ii_emb in range(len(list_feature_flatten_val)):
            print('calculate', ii, 'run time', ii_emb, 'embedding', len(list_feature_flatten_val), 'total')

            x_batch = np.expand_dims(scaler.transform(list_feature_flatten_val[ii_emb]), axis=0)
            embeddings[ii_emb, :] = model_1_batch.predict_on_batch(x_batch)

        # dist_mat = distance_matrix_embedding_classifier(embeddings)

        dist_mat = (2.0 - squareform(pdist(embeddings, 'cosine')))/2.0
        gt_mat = ground_truth_matrix(label_integer_val)

        np.save(file=os.path.join(path_eval, 'dist_mat_' + 'teacher_student_' + str(ii)), arr=dist_mat)

        ap = eval_embeddings(dist_mat=dist_mat, gt_mat=gt_mat)

        list_ap.append(ap)

    post_fix = '_teacher_student' if val_test == 'val' else '_teacher_extra_student'

    filename_eval = os.path.join(path_eval, model_name + post_fix + '.csv')

    with open(filename_eval, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', )
        csvwriter.writerow([np.mean(list_ap), np.std(list_ap)])


def embedding_frame_ap(filename_feature_teacher,
                       filename_list_key_teacher,
                       filename_feature_student,
                       filename_list_key_student,
                       filename_scaler,
                       val_test):
    """frame_leval embedding average precision"""
    logger = logging.getLogger(__name__)

    list_feature_teacher = pickle.load(open(filename_feature_teacher, 'rb'))
    list_key_teacher = pickle.load(open(filename_list_key_teacher, 'rb'))
    list_feature_student = pickle.load(open(filename_feature_student, 'rb'))
    list_key_student = pickle.load(open(filename_list_key_student, 'rb'))
    scaler = pickle.load(open(filename_scaler, 'rb'))

    list_ap = []
    embedding_dim = 54

    array_feature_replicated_teacher, array_labels_teacher, labels_teacher = \
        feature_replication_teacher_student(list_feature=list_feature_teacher,
                                            list_key=list_key_teacher,
                                            scaler=scaler,
                                            data_str='_teacher')

    array_feature_replicated_student, array_labels_student, labels_student = \
        feature_replication_teacher_student(list_feature=list_feature_student,
                                            list_key=list_key_student,
                                            scaler=scaler,
                                            data_str='_student')

    array_feature_replicated = array_feature_replicated_teacher + array_feature_replicated_student

    labels = labels_teacher + labels_student

    for ii, feature in enumerate(array_feature_replicated):
        array_feature_replicated[ii] = featureReshape(feature, nlen=7)

    path_model = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/models/phoneme_embedding_frame_level'
    path_eval = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phoneme_embedding_frame_level'
    model_name = 'wide_frame_level_emb_teacher_student'

    for ii in range(5):
        filename_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
        model = load_model(filepath=filename_model)

        embeddings = np.zeros((len(array_feature_replicated), embedding_dim))
        for ii_emb, feature in enumerate(array_feature_replicated):
            logger.info('calculating..., %s, total, %s, round, %s', ii_emb, len(array_feature_replicated), ii)

            feature = np.expand_dims(feature, axis=1)
            y_pred = model.predict_on_batch(feature)
            embeddings[ii_emb, :] = np.mean(y_pred, axis=0)

        dist_mat = (2.0 - squareform(pdist(embeddings, 'cosine')))/2.0
        gt_mat = ground_truth_matrix(labels)

        np.save(file=os.path.join(path_eval, 'dist_mat_'+str(ii)), arr=dist_mat)

        ap = eval_embeddings(dist_mat=dist_mat, gt_mat=gt_mat)

        list_ap.append(ap)

    post_fix = '' if val_test == 'val' else '_extra'

    filename_eval = os.path.join(path_eval, model_name + post_fix + '.csv')

    with open(filename_eval, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', )
        csvwriter.writerow([np.mean(list_ap), np.std(list_ap)])


def embedding_siamese_ap(filename_feature_teacher,
                         filename_list_key_teacher,
                         filename_feature_student,
                         filename_list_key_student,
                         filename_scaler,
                         model_name,
                         val_test):
    """calculate average precision of siamese triplet embedding"""

    list_feature_flatten_test, label_integer_test, le, scaler = \
        load_data_embedding_teacher_student(filename_feature_teacher,
                                            filename_list_key_teacher,
                                            filename_feature_student,
                                            filename_list_key_student,
                                            filename_scaler)

    path_model = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/models/phoneme_embedding_siamese_triplet'
    path_eval = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phoneme_embedding_siamese_triplet'

    list_ap = []
    embedding_dim = 54

    for ii in range(5):
        filename_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
        model = load_model(filepath=filename_model, compile=False)
        model_embedding = model.get_layer('embedding')
        weights = model_embedding.get_weights()
        model_embedding = embedding_base_model(input_shape=(1, None, 80),
                                               output_shape=54,
                                               base_model=embedding_2_lstm_1_dense_base)
        model_embedding.set_weights(weights=weights)

        embeddings = np.zeros((len(list_feature_flatten_test), embedding_dim))
        for ii_emb in range(len(list_feature_flatten_test)):
            print('calculate', ii, 'run time', ii_emb, 'embedding', len(list_feature_flatten_test), 'total')

            x_batch = np.expand_dims(scaler.transform(list_feature_flatten_test[ii_emb]), axis=0)
            embeddings[ii_emb, :] = model_embedding.predict_on_batch(x_batch)

        # dist_mat = distance_matrix_embedding_classifier(embeddings)

        dist_mat = (2.0 - squareform(pdist(embeddings, 'cosine')))/2.0
        gt_mat = ground_truth_matrix(label_integer_test)

        np.save(file=os.path.join(path_eval, 'dist_mat_' + str(ii)), arr=dist_mat)

        ap = eval_embeddings(dist_mat=dist_mat, gt_mat=gt_mat)

        list_ap.append(ap)

    post_fix = '' if val_test == 'val' else '_extra'

    filename_eval = os.path.join(path_eval, model_name + post_fix + '.csv')

    with open(filename_eval, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', )
        csvwriter.writerow([np.mean(list_ap), np.std(list_ap)])


if __name__ == '__main__':
    val_test = 'test'

    path_dataset = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/RongGong/phoneEmbedding'

    if val_test == 'val':
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
    filename_scaler = os.path.join(path_dataset, 'scaler_phn_embedding_train_teacher_student.pkl')

    # configs = [[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]]
    if val_test == 'test':
        embedding_classifier_ap(filename_feature_teacher,
                                filename_list_key_teacher,
                                filename_feature_student,
                                filename_list_key_student,
                                filename_scaler,
                                config=[2, 1],
                                val_test='test')
        #
        # embedding_frame_ap(filename_feature_teacher,
        #                    filename_list_key_teacher,
        #                    filename_feature_student,
        #                    filename_list_key_student,
        #                    filename_scaler,
        #                    val_test='test')

        # # for margin in ['0.15', '0.30', '0.45', '0.60', '0.75', '0.90']:
        # model_name = "phone_embedding_RNN_triplet_Ndiff5_teacher_student_margin_batch_512_cpu0.15"
        # embedding_siamese_ap(filename_feature_teacher,
        #                      filename_list_key_teacher,
        #                      filename_feature_student,
        #                      filename_list_key_student,
        #                      filename_scaler,
        #                      model_name,
        #                      val_test='test')
