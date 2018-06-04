import os
import csv
import numpy as np
import pickle
import logging
from src.distance_measures import gau_bh
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture
from training_scripts.data_preparation import load_data_embedding
from keras.models import load_model
from keras.models import Model
from training_scripts.models_RNN import model_select
from src.parameters import config_select
from training_scripts.data_preparation import feature_replication
from src.audio_preprocessing import featureReshape


def distance_matrix_embedding_gaussian(embeddings):
    """
    bhattacharyya dist mat of the guassian embeddings
    :param embeddings:
    :return:
    """
    sample_num = len(embeddings)

    distance_mat = np.zeros((sample_num, sample_num))

    for ii in range(sample_num-1):
        for jj in range(ii+1, sample_num):
            distance_mat[ii, jj] = gau_bh(embeddings[ii][0], embeddings[ii][1],
                                          embeddings[jj][0], embeddings[jj][1])

            # print(distance_mat[ii, jj])

    return distance_mat


def ground_truth_matrix(y_test):
    """
    ground truth mat
    :param y_test:
    :return:
    """
    sample_num = len(y_test)

    gt_matrix = np.zeros((sample_num, sample_num))

    for ii in range(sample_num-1):
        for jj in range(ii+1, sample_num):
            if y_test[ii] == y_test[jj]:
                gt_matrix[ii, jj] = 1.0
            else:
                gt_matrix[ii, jj] = 0.0
    return gt_matrix


def eval_embeddings(dist_mat, gt_mat):
    """
    average precision score
    :param dist_mat:
    :param gt_mat:
    :return:
    """
    assert dist_mat.shape == gt_mat.shape
    sample_num = dist_mat.shape[0]
    iu1 = np.triu_indices(sample_num, 1) # trim the upper mat

    print(len(gt_mat[iu1][gt_mat[iu1]==0]))
    print(len(gt_mat[iu1]))
    print(dist_mat[iu1])
    ap = average_precision_score(y_true=np.abs(gt_mat[iu1]), y_score=np.abs(dist_mat[iu1]), average='weighted')
    return ap


def eval_embeddings_no_trim(dist_mat, gt_mat):
    """
    average precision score
    :param dist_mat:
    :param gt_mat:
    :return:
    """
    assert dist_mat.shape == gt_mat.shape
    ap = average_precision_score(y_true=np.squeeze(np.abs(gt_mat)),
                                 y_score=np.squeeze(np.abs(dist_mat)),
                                 average='weighted')
    return ap


def embedding_classifier_ap(filename_feature, filename_list_key, filename_scaler):
    """calculate average precision of classifier embedding"""

    list_feature_flatten_test, label_integer_test, le, scaler = \
        load_data_embedding(filename_feature=filename_feature,
                            filename_list_key=filename_list_key,
                            filename_scaler=filename_scaler)

    path_model = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/models/phone_embedding_classifier'
    path_eval = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phone_embedding_classifier'

    # configs = [[1, 0], [1, 1], [2, 0], [2, 1], [2, 2], [3, 0], [3, 1], [3, 2], [3, 3]]
    configs = [[1, 1]]

    for config in configs:
        model_name = config_select(config)

        list_ap = []
        embedding_dim = 29

        for ii in range(5):
            filename_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
            model = load_model(filepath=filename_model)
            weights = model.get_weights()

            input_shape = [1, None, 80]
            model_1_batch = model_select(config=config, input_shape=input_shape)
            model_1_batch.compile(optimizer='adam',
                                  loss='categorical_crossentropy',
                                  metrics=['accuracy'])
            model_1_batch.set_weights(weights=weights)

            embeddings = np.zeros((len(list_feature_flatten_test), embedding_dim))
            for ii_emb in range(len(list_feature_flatten_test)):
                print('calculate', ii, 'run time', ii_emb, 'embedding', len(list_feature_flatten_test), 'total')

                x_batch = np.expand_dims(scaler.transform(list_feature_flatten_test[ii_emb]), axis=0)
                embeddings[ii_emb, :] = model_1_batch.predict_on_batch(x_batch)

            # dist_mat = distance_matrix_embedding_classifier(embeddings)

            dist_mat = (2.0 - squareform(pdist(embeddings, 'cosine')))/2.0
            gt_mat = ground_truth_matrix(label_integer_test)

            np.save(file=os.path.join(path_eval, 'dist_mat_' + str(ii)), arr=dist_mat)

            ap = eval_embeddings(dist_mat=dist_mat, gt_mat=gt_mat)

            list_ap.append(ap)

        filename_eval = os.path.join(path_eval, model_name + '.csv')
        with open(filename_eval, 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', )
            csvwriter.writerow([np.mean(list_ap), np.std(list_ap)])


def embedding_siamese_ap(filename_feature, filename_list_key, filename_scaler, model_name):
    """calculate average precision of siamese triplet embedding"""

    list_feature_flatten_test, label_integer_test, le, scaler = \
        load_data_embedding(filename_feature=filename_feature,
                            filename_list_key=filename_list_key,
                            filename_scaler=filename_scaler)

    path_model = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/models/phoneme_embedding_siamese_triplet'
    path_eval = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phoneme_embedding_siamese_triplet'

    list_ap = []
    embedding_dim = 29

    for ii in range(5):
        filename_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
        model = load_model(filepath=filename_model, compile=False)
        model_embedding = model.get_layer('embedding')

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

    filename_eval = os.path.join(path_eval, model_name + '.csv')
    with open(filename_eval, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', )
        csvwriter.writerow([np.mean(list_ap), np.std(list_ap)])


def embedding_frame_ap(filename_feature, filename_list_key, filename_scaler):
    """frame_leval embedding average precision"""
    logger = logging.getLogger(__name__)

    list_feature = pickle.load(open(filename_feature, 'rb'))
    list_key = pickle.load(open(filename_list_key, 'rb'))
    scaler = pickle.load(open(filename_scaler, 'rb'))

    list_ap = []
    embedding_dim = 29

    array_feature_replicated, array_labels, labels = \
        feature_replication(list_feature=list_feature, list_key=list_key, scaler=scaler)

    for ii, feature in enumerate(array_feature_replicated):
        array_feature_replicated[ii] = featureReshape(feature, nlen=7)

    path_model = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/models/phoneme_embedding_frame_level'
    path_eval = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phoneme_embedding_frame_level'
    model_name = 'wide_frame_level_emb'

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

    filename_eval = os.path.join(path_eval, model_name + '.csv')
    with open(filename_eval, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', )
        csvwriter.writerow([np.mean(list_ap), np.std(list_ap)])


def gaussin_embedding_ap(filename_feature, filename_list_key, filename_scaler):
    """I wanted to use the gaussian as the embedding,
    but Bhattacharyya distance may not in [0, 1], so I normalized them by dividing the maximum value """

    logger = logging.getLogger(__name__)

    list_feature_flatten_test, label_integer_test, le, scaler = \
        load_data_embedding(filename_feature=filename_feature,
                            filename_list_key=filename_list_key,
                            filename_scaler=filename_scaler)

    embeddings = []
    for ii_emb in range(len(list_feature_flatten_test)):
        logger.info('calculating..., %s, total, %s', ii_emb, len(list_feature_flatten_test))

        feature = scaler.transform(list_feature_flatten_test[ii_emb])

        # print(feature.shape)
        gmm = GaussianMixture(n_components=1, covariance_type='diag')
        gmm.fit(feature)
        mu = gmm.means_[0, :]
        cov = gmm.covariances_[0, :]
        embeddings.append([mu, cov])

    dist_mat = distance_matrix_embedding_gaussian(embeddings)
    dist_mat = 1.0 - dist_mat/np.max(dist_mat)
    gt_mat = ground_truth_matrix(label_integer_test)

    ap = eval_embeddings(dist_mat=dist_mat, gt_mat=gt_mat)
    return ap


def correlationDistanceMat():
    """calculate the corr coef between two distance mats"""
    path_eval_0 = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phoneme_embedding_frame_level'
    path_eval_1 = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phone_embedding_classifier'
    path_eval_2 = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phoneme_embedding_siamese_triplet'

    dist_mat_0 = np.load(os.path.join(path_eval_0, 'dist_mat_0.npy'))
    dist_mat_1 = np.load(os.path.join(path_eval_1, 'dist_mat_0.npy'))
    dist_mat_2 = np.load(os.path.join(path_eval_2, 'dist_mat_0.npy'))

    sample_num = dist_mat_0.shape[0]
    iu1 = np.triu_indices(sample_num, 1)  # trim the upper mat

    r_01, _ = spearmanr(dist_mat_0[iu1], dist_mat_1[iu1])
    r_02, _ = spearmanr(dist_mat_0[iu1], dist_mat_2[iu1])
    r_12, _ = spearmanr(dist_mat_1[iu1], dist_mat_2[iu1])

    print('correlation frame, classifier', r_01)
    print('correlation frame, siamese', r_02)
    print('correlation classifier, siamese', r_12)
    # return r


if __name__ == '__main__':
    # teacher test set
    filename_feature = '/home/gong/Documents/MTG/dataset/phoneEmbedding/feature_phn_embedding_test.pkl'
    filename_list_key = '/home/gong/Documents/MTG/dataset/phoneEmbedding/list_key.pkl'
    filename_scaler = '/home/gong/Documents/MTG/dataset/phoneEmbedding/scaler_phn_embedding.pkl'

    for model_name in ['phone_embedding_RNN_triplet_pairs_margin15',
                       'phone_embedding_RNN_triplet_margin15_reverse_anchor']:
        embedding_siamese_ap(filename_feature, filename_list_key, filename_scaler, model_name)

    # correlationDistanceMat()

