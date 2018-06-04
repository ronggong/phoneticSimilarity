import csv
import os
import pickle
import logging
import numpy as np
import pandas as pd
from eval_embedding import ground_truth_matrix
# from eval_embedding import eval_embeddings
from training_scripts.data_preparation import load_data_embedding_teacher_student
from training_scripts.data_preparation import feature_replication_teacher_student
from src.parameters import config_select
from training_scripts.models_RNN import model_select
from training_scripts.models_RNN import model_select_attention
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from src.audio_preprocessing import featureReshape
from sklearn.metrics import average_precision_score
from src.utilFunctions import get_unique_label
from training_scripts.attention import Attention
from training_scripts.attentionWithContext import AttentionWithContext

from training_scripts.models_siamese_tripletloss import embedding_1_lstm_base
from training_scripts.models_siamese_tripletloss import embedding_2_lstm_1_dense_base
from training_scripts.models_siamese_tripletloss import embedding_base_model


def embedding_classifier_ap(filename_feature_teacher,
                            filename_list_key_teacher,
                            filename_feature_student,
                            filename_list_key_student,
                            filename_scaler,
                            embedding_dim,
                            config,
                            val_test,
                            MTL=False,
                            attention=False,
                            dense=False,
                            conv=False,
                            dropout=0.25):
    """calculate teacher student pairs average precision of classifier embedding"""

    list_feature_flatten_val, label_integer_val, le, scaler = \
        load_data_embedding_teacher_student(filename_feature_teacher=filename_feature_teacher,
                                            filename_list_key_teacher=filename_list_key_teacher,
                                            filename_feature_student=filename_feature_student,
                                            filename_list_key_student=filename_list_key_student,
                                            filename_scaler=filename_scaler)

    path_model = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/models/phone_embedding_classifier'
    path_eval = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phone_embedding_classifier'

    # configs = [[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]]
    # configs = [[2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]]

    prefix = '_MTL' if MTL else '_2_class_teacher_student'
    model_name = config_select(config) + prefix if embedding_dim == 2 or embedding_dim == 32 else config_select(config)
    if dense and conv:
        attention_dense_str = 'dense_conv_'
    elif attention:
        attention_dense_str = "attention_"
    elif dense:
        attention_dense_str = "dense_"
    elif conv:
        attention_dense_str = "conv_"
    elif dropout:
        attention_dense_str = "dropout_"
    else:
        attention_dense_str = ""

    list_ap = []
    # average precision of each phone
    array_ap_phn_5_runs = np.zeros((5, 27))
    for ii in range(5):
        print('run time', ii)
        filename_model = os.path.join(path_model, model_name + '_' + attention_dense_str + str(ii) + '.h5')
        if attention:
            model = load_model(filepath=filename_model, custom_objects={'Attention': Attention(return_attention=True)})
        else:
            model = load_model(filepath=filename_model)
        weights = model.get_weights()

        input_shape = [1, None, 80]
        if attention:
            x, input, _ = model_select_attention(config=config, input_shape=input_shape, conv=conv, dropout=dropout)
        else:
            x, input = model_select(config=config, input_shape=input_shape, conv=conv, dropout=dropout)

        if MTL:
            if dense:
                pronun_out = x
                profess_out = Dense(32)(x)
            else:
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

        embeddings = np.zeros((len(list_feature_flatten_val), embedding_dim))
        for ii_emb in range(len(list_feature_flatten_val)):
            # print('calculate', ii, 'run time', ii_emb, 'embedding', len(list_feature_flatten_val), 'total')

            x_batch = np.expand_dims(scaler.transform(list_feature_flatten_val[ii_emb]), axis=0)
            if MTL:
                _, out = model_1_batch.predict_on_batch(x_batch)
            else:
                out = model_1_batch.predict_on_batch(x_batch)

            if attention:
                embeddings[ii_emb, :] = out[0, :]
            else:
                embeddings[ii_emb, :] = out

        # dist_mat = distance_matrix_embedding_classifier(embeddings)

        list_dist = []
        list_gt = []
        array_ap_phn = np.zeros((27,))
        cols = []
        list_ratio_tea_stu = []
        for ii_class in range(27):
            # teacher student pair class index
            idx_ii_class = np.where(np.logical_or(label_integer_val == 2*ii_class,
                                                  label_integer_val == 2*ii_class+1))[0]

            idx_ii_class_stu = len(np.where(label_integer_val == 2*ii_class)[0])
            idx_ii_class_tea = len(np.where(label_integer_val == 2*ii_class+1)[0])

            # ratio of teacher's samples
            list_ratio_tea_stu.append(idx_ii_class_tea/float(idx_ii_class_tea+idx_ii_class_stu))

            dist_mat = (2.0 - squareform(pdist(embeddings[idx_ii_class], 'cosine')))/2.0
            labels_ii_class = [label_integer_val[idx] for idx in idx_ii_class]
            gt_mat = ground_truth_matrix(labels_ii_class)

            sample_num = dist_mat.shape[0]
            iu1 = np.triu_indices(sample_num, 1)  # trim the upper mat

            list_dist.append(dist_mat[iu1])
            list_gt.append(gt_mat[iu1])

            # calculate the average precision of each phoneme
            ap_phn = average_precision_score(y_true=np.abs(list_gt[ii_class]),
                                             y_score=np.abs(list_dist[ii_class]),
                                             average='weighted')

            cols.append(le.inverse_transform(2*ii_class).split('_')[0])
            array_ap_phn[ii_class] = ap_phn

            print(list_ratio_tea_stu)

        array_dist = np.concatenate(list_dist)
        array_gt = np.concatenate(list_gt)

        ap = average_precision_score(y_true=np.abs(array_gt), y_score=np.abs(array_dist), average='weighted')

        list_ap.append(ap)

        array_ap_phn_5_runs[ii, :] = array_ap_phn

    post_fix = prefix+'_2_class' if val_test == 'val' else prefix+'_2_class_extra_student'

    filename_eval = os.path.join(path_eval, model_name + post_fix + attention_dense_str + '.csv')

    with open(filename_eval, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', )
        csvwriter.writerow([np.mean(list_ap), np.std(list_ap)])

    # organize the Dataframe
    ap_phn_mean = np.mean(array_ap_phn_5_runs, axis=0)
    ap_phn_std = np.std(array_ap_phn_5_runs, axis=0)
    ap_phn_mean_std = pd.DataFrame(np.transpose(np.vstack((ap_phn_mean, ap_phn_std, list_ratio_tea_stu))),
                                   columns=['mean', 'std', 'ratio'],
                                   index=cols)

    ap_phn_mean_std = ap_phn_mean_std.sort_values(by='mean')
    ap_phn_mean_std.to_csv(os.path.join(path_eval,
                                        model_name + post_fix + attention_dense_str + '_phn_mean_std.csv'))


def embedding_frame_ap(filename_feature_teacher,
                       filename_list_key_teacher,
                       filename_feature_student,
                       filename_list_key_student,
                       filename_scaler,
                       embedding_dim,
                       val_test):
    """frame_leval embedding average precision"""
    logger = logging.getLogger(__name__)

    list_feature_teacher = pickle.load(open(filename_feature_teacher, 'rb'))
    list_key_teacher = pickle.load(open(filename_list_key_teacher, 'rb'))
    list_feature_student = pickle.load(open(filename_feature_student, 'rb'))
    list_key_student = pickle.load(open(filename_list_key_student, 'rb'))
    scaler = pickle.load(open(filename_scaler, 'rb'))

    list_ap = []

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

    labels = np.array(labels_teacher + labels_student)

    for ii, feature in enumerate(array_feature_replicated):
        array_feature_replicated[ii] = featureReshape(feature, nlen=7)

    path_model = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/models/phoneme_embedding_frame_level'
    path_eval = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phoneme_embedding_frame_level'

    model_name = 'wide_frame_level_emb_teacher_student_2_class' if embedding_dim == 2 \
        else 'wide_frame_level_emb_teacher_student'

    for ii in range(5):
        filename_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
        model = load_model(filepath=filename_model)

        embeddings = np.zeros((len(array_feature_replicated), embedding_dim))
        for ii_emb, feature in enumerate(array_feature_replicated):
            logger.info('calculating..., %s, total, %s, round, %s', ii_emb, len(array_feature_replicated), ii)

            feature = np.expand_dims(feature, axis=1)
            y_pred = model.predict_on_batch(feature)
            embeddings[ii_emb, :] = np.mean(y_pred, axis=0)

        list_dist = []
        list_gt = []
        for ii_class in range(27):
            idx_ii_class = \
            np.where(np.logical_or(labels == ii_class, labels == ii_class + 27))[0]
            dist_mat = (2.0 - squareform(pdist(embeddings[idx_ii_class, :], 'cosine')))/2.0
            labels_ii_class = [labels[idx] for idx in idx_ii_class]
            gt_mat = ground_truth_matrix(labels_ii_class)

            # np.save(file=os.path.join(path_eval, 'dist_mat_'+str(ii)), arr=dist_mat)

            sample_num = dist_mat.shape[0]
            iu1 = np.triu_indices(sample_num, 1)  # trim the upper mat

            list_dist.append(dist_mat[iu1])
            list_gt.append(gt_mat[iu1])

        list_dist = np.concatenate(list_dist)
        list_gt = np.concatenate(list_gt)

        ap = average_precision_score(y_true=np.abs(list_gt), y_score=np.abs(list_dist), average='weighted')

        list_ap.append(ap)

    post_fix = '_pairs' if val_test == 'val' else '_extra_pairs'

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
                         embedding_dim,
                         val_test):
    """calculate average precision of siamese triplet embedding"""

    list_feature_flatten_test, labels, le, scaler = \
        load_data_embedding_teacher_student(filename_feature_teacher,
                                            filename_list_key_teacher,
                                            filename_feature_student,
                                            filename_list_key_student,
                                            filename_scaler)

    path_model = './models/phoneme_embedding_siamese_triplet'
    path_eval = './eval/phoneme_embedding_siamese_triplet'

    labels_unique = get_unique_label(labels)
    labels_unique_str = le.inverse_transform(labels_unique)

    list_ap = []

    for ii in range(5):
        filename_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
        model = load_model(filepath=filename_model, compile=False)
        model_embedding = model.get_layer('embedding')
        weights = model_embedding.get_weights()

        if embedding_dim == 2:
            embedding_model = embedding_1_lstm_base
        else:
            embedding_model = embedding_2_lstm_1_dense_base

        model_embedding = embedding_base_model(input_shape=(1, None, 80),
                                               output_shape=embedding_dim,
                                               base_model=embedding_model)
        model_embedding.set_weights(weights=weights)

        embeddings = np.zeros((len(list_feature_flatten_test), embedding_dim))
        for ii_emb in range(len(list_feature_flatten_test)):
            print('calculate', ii, 'run time', ii_emb, 'embedding', len(list_feature_flatten_test), 'total')

            x_batch = np.expand_dims(scaler.transform(list_feature_flatten_test[ii_emb]), axis=0)
            embeddings[ii_emb, :] = model_embedding.predict_on_batch(x_batch)

        list_dist = []
        list_gt = []
        for ii_teacher in range(27):
            # teacher class index
            ii_class_teacher = labels_unique[ii_teacher]

            # phoneme teacher
            phn = labels_unique_str[ii_teacher].split('_')[0]

            ii_student = np.where(labels_unique_str == phn+'_student')[0][0]

            # student class index
            ii_class_student = labels_unique[ii_student]

            idx_ii_class = \
                np.where(np.logical_or(labels == ii_class_teacher, labels == ii_class_student))[0]
            dist_mat = (2.0 - squareform(pdist(embeddings[idx_ii_class, :], 'cosine'))) / 2.0
            labels_ii_class = [labels[idx] for idx in idx_ii_class]
            gt_mat = ground_truth_matrix(labels_ii_class)

            # np.save(file=os.path.join(path_eval, 'dist_mat_'+str(ii)), arr=dist_mat)

            sample_num = dist_mat.shape[0]
            iu1 = np.triu_indices(sample_num, 1)  # trim the upper mat

            list_dist.append(dist_mat[iu1])
            list_gt.append(gt_mat[iu1])

        list_dist = np.concatenate(list_dist)
        list_gt = np.concatenate(list_gt)

        ap = average_precision_score(y_true=np.abs(list_gt), y_score=np.abs(list_dist), average='weighted')

        list_ap.append(ap)

    post_fix = '_pairs' if val_test == 'val' else '_extra_pairs'

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
        # embedding_classifier_ap(filename_feature_teacher,
        #                         filename_list_key_teacher,
        #                         filename_feature_student,
        #                         filename_list_key_student,
        #                         filename_scaler,
        #                         embedding_dim=54,
        #                         config=[2, 1],
        #                         val_test=val_test)
        #
        embedding_classifier_ap(filename_feature_teacher,
                                filename_list_key_teacher,
                                filename_feature_student,
                                filename_list_key_student,
                                filename_scaler,
                                embedding_dim=32,
                                config=[1, 0],
                                val_test=val_test,
                                MTL=False,
                                attention=False,
                                dense=True,
                                conv=True,
                                dropout=False)

        # embedding_frame_ap(filename_feature_teacher,
        #                    filename_list_key_teacher,
        #                    filename_feature_student,
        #                    filename_list_key_student,
        #                    filename_scaler,
        #                    embedding_dim=54,
        #                    val_test=val_test)
        #
        # embedding_frame_ap(filename_feature_teacher,
        #                    filename_list_key_teacher,
        #                    filename_feature_student,
        #                    filename_list_key_student,
        #                    filename_scaler,
        #                    embedding_dim=2,
        #                    val_test=val_test)

        # for margin in ['0.0', '0.15', '0.30', '0.45', '0.60', '0.75', '0.90']:
        # model_name = "phone_embedding_RNN_triplet_Ndiff5_teacher_student_margin_batch_512_cpu0.15"
        # embedding_siamese_ap(filename_feature_teacher,
        #                      filename_list_key_teacher,
        #                      filename_feature_student,
        #                      filename_list_key_student,
        #                      filename_scaler,
        #                      model_name,
        #                      embedding_dim=54,
        #                      val_test='test')

        # model_name = "phone_embedding_RNN_triplet_Ndiff5_teacher_student_margin_2_class_batch_512_cpu0.45"
        # embedding_siamese_ap(filename_feature_teacher,
        #                      filename_list_key_teacher,
        #                      filename_feature_student,
        #                      filename_list_key_student,
        #                      filename_scaler,
        #                      model_name,
        #                      embedding_dim=2,
        #                      val_test='test')
    else:
        # configs = [[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]]
        configs = [[1, 0]]
        for config in configs:
            embedding_classifier_ap(filename_feature_teacher,
                                    filename_list_key_teacher,
                                    filename_feature_student,
                                    filename_list_key_student,
                                    filename_scaler,
                                    embedding_dim=2,
                                    config=config,
                                    val_test=val_test,
                                    MTL=False,
                                    attention=True)
