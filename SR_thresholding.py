"""
idea is to threshold the structural representation scores of each phone,
obtain the bad phones, penalize the machine rating score
"""

import matplotlib
matplotlib.use('TkAgg')

import os
import pickle
import json
import numpy as np
import scipy
import matplotlib.pyplot as plt

from eval import adjustHumanRatingJson
from eval import humanMachineCorrCoef


def histoDistortion(path_feature):

    # total span: [4, 100]
    # head span: [4, 111]
    # belly span: [2, 46]

    for mode in ['_total' , '_head', '_belly']:
        features = pickle.load(open(os.path.join(path_feature, 'SR_oracle_val'+mode+'.pkl'), 'rb'))

        distortion_phns_all = np.array([])
        for artist in features:
            for aria in features[artist]:
                distortion_phns = features[artist][aria]['distortion_phns']
                distortion_phns_all = np.concatenate((distortion_phns_all, distortion_phns))

        plt.figure()
        plt.hist(distortion_phns_all, 1000)
        plt.title(mode)
        plt.show()


def ratingByThresholding(features, th=10.0):
    """
    every phn distortion > th will result in 5 points diminish in the rating of the line
    :param features:
    :param th:
    :return:
    """
    rating = {}
    for artist in features:
        rating[artist] = {}
        for aria in features[artist]:
            distortion_phns = features[artist][aria]['distortion_phns']
            phns_bad = distortion_phns[distortion_phns>th]
            score = 100.0 - len(phns_bad)*5
            score = score if score >= 0.0 else 0.0

            rating[artist][aria] = score

    return rating

if __name__ == '__main__':
    path_feature = './data/training_features'

    # histoDistortion(path_feature=path_feature)

    path_data = './data'
    rating_human = json.load(open(os.path.join(path_data, 'rating.json'), 'r'))
    rating_human_adjusted = adjustHumanRatingJson(rating_human)

    coef_val = {}
    for mode in ['_total' , '_head', '_belly']:
        features = pickle.load(open(os.path.join(path_feature, 'SR_oracle_val'+mode+'.pkl'), 'rb'))
        r_best = 0.0
        th_best = 0.0
        for th in range(1, 1000):
            rating_machine = ratingByThresholding(features, th=th*0.1)
            print(th*0.1, rating_machine)
            r = humanMachineCorrCoef(rating_human=rating_human_adjusted, rating_machine=rating_machine)
            if np.abs(r) > r_best:
                th_best = th*0.1
                r_best = r
        coef_val[mode] = [r_best, th_best]

    print(coef_val)
