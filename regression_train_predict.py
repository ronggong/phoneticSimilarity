"""
idea is to use regression model to predict the human rating
"""
import os
import pickle
import json
from eval import adjustHumanRatingJson
from sklearn.svm import SVR

import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import spearmanr

from sklearn import linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras.losses import mean_absolute_error
from keras import backend as K
from models_RNN_regression import embedding_RNN


def phnLevelStatistics(x):
    # feature = [np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x), skew(x), kurtosis(x)]
    feature = [np.mean(x), np.std(x), skew(x), kurtosis(x)]

    return feature

def organizeFeaturesStat(features_head, features_belly):
    X, y = [], []
    for artist in features_head:
        for aria in features_head[artist]:
            score_head = features_head[artist][aria]['distortion_phns']
            score_belly = features_belly[artist][aria]['distortion_phns']
            num_tails_missing = features_head[artist][aria]['num_tails_missing']

            try:
                x_aria = phnLevelStatistics(score_head) + phnLevelStatistics(score_belly)
                # num_tails_missing]

                y_aria = rating_human_adjusted[artist.lower()][aria]
                X.append(x_aria)
                y.append(y_aria)
            except KeyError:
                print(artist, aria, 'not found')

    X = np.array(X)
    y = np.array(y)
    return X, y


def organizeFeaturesTotal(features_total):
    X, y = [], []
    for artist in features_total:
        for aria in features_total[artist]:
            score_head = features_total[artist][aria]['distortion_phns']
            num_tails_missing = features_total[artist][aria]['num_tails_missing']

            try:
                y_aria = rating_human_adjusted[artist.lower()][aria]
                X.append(score_head.reshape(1, -1))
                y.append(y_aria)
            except KeyError:
                print(artist, aria, 'not found')

    y = np.array(y)/100.0
    return X, y


def statisticsTrainPredict(features_head_train,
                           features_belly_train,
                           features_head_test,
                           features_belly_test,
                           mode):
    X_val, y_val = organizeFeaturesStat(features_head=features_head_train,
                                        features_belly=features_belly_train)

    reg = linear_model.LinearRegression()
    reg.fit(X_val, y_val)

    ransac = linear_model.RANSACRegressor()
    ransac.fit(X_val, y_val)

    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(X_val, y_val)

    X_test, y_test = organizeFeaturesStat(features_head=features_head_test,
                                          features_belly=features_belly_test)

    y_pred_lin = reg.predict(X_test)
    y_pred_ran = ransac.predict(X_test)
    y_pred_svr = svr_rbf.predict(X_test)

    r, p = spearmanr(y_test, y_pred_lin)
    print(r)

    y_pred_lin = reg.predict(X_val)
    r, p = spearmanr(y_val, y_pred_lin)
    print(r)


def model_prediction(model, X, scaler):
    """
    scaling first, then predict
    :param model:
    :param X:
    :param scaler:
    :return:
    """
    y_pred = np.zeros((len(X), ))
    for ii in range(len(X)):
        X_sample = np.expand_dims(scaler.transform(X[ii].T), axis=0)
        y_pred[ii] = model.predict_on_batch(X_sample)
    return y_pred


def evaluate_model(model, X, y, scaler):
    """
    eval the model with mse loss
    :param model:
    :param X:
    :param y:
    :param scaler:
    :return:
    """

    y_pred = model_prediction(model, X, scaler)

    # print(y.shape, y_pred.shape)
    y = K.variable(y)
    y_pred = K.variable(y_pred)

    loss = K.eval(mean_absolute_error(y, y_pred))

    return loss


def shuffleFeaturesLabelsInUnison(features, labels):
    """
    shuffle features and labels each epoch
    :param features:
    :param labels:
    :return:
    """
    p=np.random.permutation(len(features))
    features = [features[ii] for ii in p]
    return features, labels[p]


def LSTMTrain(model,
              X_train,
              y_train,
              X_val,
              y_val,
              scaler):

    nb_epochs = 500
    best_val_loss = np.inf  # initialize the val_loss
    counter = 0
    patience = 15  # early stopping patience
    best_model = None

    for ii_epoch in range(nb_epochs):
        for ii in range(len(X_train)):
            X_train_sample = np.expand_dims(scaler.transform(X_train[ii].T), axis=0)
            y_train_sample = np.expand_dims(y_train[ii], axis=0)
            results = model.train_on_batch(X_train_sample, y_train_sample)

            # print(results)

        train_loss = evaluate_model(model=model, X=X_train, y=y_train, scaler=scaler)
        val_loss = evaluate_model(model=model, X=X_val, y=y_val, scaler=scaler)

        print(ii_epoch, train_loss, val_loss)

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # model.save_weights(file_path_model)
            best_model = model
        else:
            counter += 1

        # # write validation loss to csv
        # writeValLossCsv(file_path_log=file_path_log,
        #                 ii_epoch=ii_epoch,
        #                 val_loss=val_loss,
        #                 train_loss=train_loss)

        # early stopping
        if counter >= patience:
            break

        X_train, y_train = shuffleFeaturesLabelsInUnison(features=X_train, labels=y_train)

    return best_model


if __name__ == '__main__':
    path_feature = './data/training_features'
    path_data = './data'

    mode = 'GOP' # GOP

    # human rating, ground truth
    rating_human = json.load(open(os.path.join(path_data, 'rating.json'), 'r'))
    rating_human_adjusted = adjustHumanRatingJson(rating_human)

    # training
    features_head_train = pickle.load(open(os.path.join(path_feature, mode+'_oracle_val_head.pkl'), 'rb'))
    features_belly_train = pickle.load(open(os.path.join(path_feature, mode+'_oracle_val_belly.pkl'), 'rb'))
    features_total_train = pickle.load(open(os.path.join(path_feature, mode+'_oracle_val_total.pkl'), 'rb'))

    # predict
    features_head_test = pickle.load(open(os.path.join(path_feature, mode + '_oracle_test_head.pkl'), 'rb'))
    features_belly_test = pickle.load(open(os.path.join(path_feature, mode + '_oracle_test_belly.pkl'), 'rb'))
    features_total_test = pickle.load(open(os.path.join(path_feature, mode + '_oracle_test_total.pkl'), 'rb'))

    X_train_val, y_train_val = organizeFeaturesTotal(features_total=features_total_train)
    X_test, y_test = organizeFeaturesTotal(features_total=features_total_test)

    # fit the scaler
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(np.hstack(X_train_val).T)

    # split train val sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=True)

    input_shape = [None, 1]

    model = embedding_RNN(input_shape)

    model = LSTMTrain(model=model,
                      X_train=X_train,
                      y_train=y_train,
                      X_val=X_val,
                      y_val=y_val,
                      scaler=scaler)

    y_pred_test = model_prediction(model=model, X=X_test, scaler=scaler)

    r, p = spearmanr(y_test, y_pred_test)

    print(r)