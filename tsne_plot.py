import matplotlib
matplotlib.use('Tkagg')

import os
# import csv
import numpy as np
import pickle
import logging
# from src.distance_measures import gau_bh
# from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from data_preparation import load_data_embedding
from keras.models import load_model
# from keras.models import Model
from models_RNN import model_select
from parameters import config_select
from data_preparation import feature_replication
from src.audio_preprocessing import featureReshape

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_tsne(embeddings, labels):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=3000)
    tsne_results = tsne.fit_transform(embeddings)
    plt.figure()
    colors = iter(cm.rainbow(np.linspace(0, 1, 29)))
    for ii_class in range(29):
        plt.scatter(tsne_results[labels==ii_class,0],
                    tsne_results[labels==ii_class,1], color=next(colors))
    plt.show()


def embedding_frame_tsne(filename_feature, filename_list_key, filename_scaler):
    """frame_leval embedding average precision"""
    logger = logging.getLogger(__name__)

    list_feature = pickle.load(open(filename_feature, 'rb'))
    list_key = pickle.load(open(filename_list_key, 'rb'))
    scaler = pickle.load(open(filename_scaler, 'rb'))

    path_model = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/models/phoneme_embedding_frame_level'
    path_eval = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phoneme_embedding_frame_level'
    model_name = 'wide_frame_level_emb'

    embedding_dim = 29

    array_feature_replicated, array_labels, labels = \
        feature_replication(list_feature=list_feature, list_key=list_key, scaler=scaler)

    for ii, feature in enumerate(array_feature_replicated):
        array_feature_replicated[ii] = featureReshape(feature, nlen=7)

    np.save(file=os.path.join(path_eval, model_name + '_labels'), arr=labels)

    for ii in range(1):
        filename_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
        model = load_model(filepath=filename_model)

        embeddings = np.zeros((len(array_feature_replicated), embedding_dim))
        for ii_emb, feature in enumerate(array_feature_replicated):
            logger.info('calculating..., %s, total, %s, round, %s', ii_emb, len(array_feature_replicated), ii)

            feature = np.expand_dims(feature, axis=1)
            y_pred = model.predict_on_batch(feature)
            embeddings[ii_emb, :] = np.mean(y_pred, axis=0)

        np.save(file=os.path.join(path_eval, model_name + '_embedding_' + str(ii)), arr=embeddings)


def embedding_classifier_tsne(filename_feature, filename_list_key, filename_scaler):
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
        pickle.dump(le, open(os.path.join(path_eval, model_name + '_le.pkl'), 'wb'), protocol=2)

        embedding_dim = 29

        for ii in range(1):
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

            np.save(file=os.path.join(path_eval, model_name + '_embedding_' + str(ii)), arr=embeddings)
            np.save(file=os.path.join(path_eval, model_name + '_labels_'), arr=label_integer_test)


def embedding_siamese_tsne(filename_feature, filename_list_key, filename_scaler):
    """calculate average precision of siamese triplet embedding"""

    list_feature_flatten_test, label_integer_test, le, scaler = \
        load_data_embedding(filename_feature=filename_feature,
                            filename_list_key=filename_list_key,
                            filename_scaler=filename_scaler)

    path_model = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/models/phoneme_embedding_siamese_triplet'
    path_eval = '/Users/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phoneme_embedding_siamese_triplet'

    model_name = 'phone_embedding_RNN_triplet_margin08'

    np.save(file=os.path.join(path_eval, model_name + '_labels'), arr=label_integer_test)
    pickle.dump(le, open(os.path.join(path_eval, model_name + '_le.pkl'), 'wb'), protocol=2)

    embedding_dim = 29

    for ii in range(1):
        filename_model = os.path.join(path_model, model_name + '_' + str(ii) + '.h5')
        model = load_model(filepath=filename_model, compile=False)
        model_embedding = model.get_layer('embedding')

        embeddings = np.zeros((len(list_feature_flatten_test), embedding_dim))
        for ii_emb in range(len(list_feature_flatten_test)):
            print('calculate', ii, 'run time', ii_emb, 'embedding', len(list_feature_flatten_test), 'total')

            x_batch = np.expand_dims(scaler.transform(list_feature_flatten_test[ii_emb]), axis=0)
            embeddings[ii_emb, :] = model_embedding.predict_on_batch(x_batch)

        np.save(file=os.path.join(path_eval, model_name + '_embedding_' + str(ii)), arr=embeddings)


if __name__ == '__main__':
    # teacher test set
    filename_feature = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/feature_phn_embedding_test.pkl'
    filename_list_key = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/list_key.pkl'
    filename_scaler = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/scaler_phn_embedding.pkl'

    # student test set
    # filename_feature = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/feature_phn_embedding_test_student.pkl'
    # filename_list_key = '/Users/gong/Documents/MTG document/dataset/phoneEmbedding/list_key_student.pkl'

    embedding_classifier_tsne(filename_feature, filename_list_key, filename_scaler)
    # embedding_siamese_tsne(filename_feature, filename_list_key, filename_scaler)
    # embedding_frame_tsne(filename_feature, filename_list_key, filename_scaler)

    # correlationDistanceMat()

