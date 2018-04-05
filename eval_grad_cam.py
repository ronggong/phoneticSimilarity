"""The RNN classification model is not working in LIME, because for the tabular data, LIME needs training data,
the format of the training dataset is [samples, times, features], each sample needs to be the same time length,
which is not our case. We have sample with different time length.

frame model is not working in LIME, because the lime_image uses skimage which needs that the image
ValueError: Images of type float must be between -1 and 1. However we can't guarantee this, because we use
StandardScaler which is not necessarily output the feature value between -1 and 1. This error should be able to avoid
by changing the condition in skimage. However, because RNN classification model is not working, we will not try this.
Because if we can't make LIME to work for all the testing models, we can't compare between them."""

# import csv
import os
import pickle
import logging
import numpy as np
# from eval_embedding import ground_truth_matrix
# # from eval_embedding import eval_embeddings
# from parameters import config_select
# from models_RNN import model_select
# from data_preparation import load_data_embedding_teacher_student
from data_preparation import feature_replication_teacher_student
from keras.models import load_model
from src.audio_preprocessing import featureReshape
# from scipy.spatial.distance import pdist
# from scipy.spatial.distance import squareform
# from sklearn.metrics import average_precision_score
# from src.utilFunctions import get_unique_label
from keras import backend as K

import matplotlib.pyplot as plt
import cv2


def grad_cam(model, x, layer_name):

    teacher_output = model.output[:, 0]

    last_conv_layer = model.get_layer(layer_name)

    # gradient of the 0th variable
    grads = K.gradients(teacher_output, last_conv_layer.output)[0]

    pooled_grads = K.mean(grads, axis=(0, 2, 3))

    iterate = K.function([model.input],
                         [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([x])

    for i in range(len(pooled_grads_value)):
        conv_layer_output_value[i, :, :] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=0)

    heatmap = np.maximum(heatmap, 0)

    return heatmap


def heatmap_postprocessing(heatmap):

    heatmap /= np.max(heatmap)

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    return heatmap


def featuremap_post_processing(feature_map):

    feature_map -= np.min(feature_map)
    feature_map /= np.max(feature_map)

    feature_map = np.uint8(255 * feature_map)

    feature_map = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)

    return feature_map


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

        print(model.summary())

        embeddings = np.zeros((len(array_feature_replicated), embedding_dim))
        for ii_emb, feature in enumerate(array_feature_replicated):
            logger.info('calculating..., %s, total, %s, round, %s', ii_emb, len(array_feature_replicated), ii)

            feature = np.expand_dims(feature, axis=1)

            print(feature.shape)

            feature_len = feature.shape[0]

            heatmap_concatenate = np.zeros((80, feature_len))

            for ii in range(feature_len):
                feature_map = feature[ii, 0]
                heatmap = grad_cam(model, feature[ii:ii+1], layer_name='conv2d_2')

                if not np.isnan(np.sum(heatmap)):
                    # print(heatmap.shape)
                    # plt.matshow(heatmap)
                    # plt.show()

                    heatmap = cv2.resize(heatmap, (15, 80))

                    heatmap_concatenate[:, ii] = heatmap[:, 7]

                    # fin = cv2.addWeighted(heatmap, 0.7, feature_map, 0.3, 0)

                    # superimposed_img = heatmap + feature_map

                    # print(heatmap*0.4)

            print(heatmap_concatenate)

            heatmap_concatenate = heatmap_postprocessing(heatmap=heatmap_concatenate)

            plt.imshow(heatmap_concatenate)

            plt.show()

                # y_pred = model.predict_on_batch(feature)


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
        embedding_frame_ap(filename_feature_teacher,
                           filename_list_key_teacher,
                           filename_feature_student,
                           filename_list_key_student,
                           filename_scaler,
                           embedding_dim=2,
                           val_test=val_test)