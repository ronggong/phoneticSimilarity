import json
import os
import numpy as np
from scipy.stats import spearmanr

def dict2score(rating_human, rating_machine):
    """
    Convert dictionary to score list
    :param rating_human:
    :param rating_machine:
    :return:
    """
    score_machine = []
    score_human = []
    for key in rating_machine:
        for key_aria in rating_machine[key]:
            try:
                score_human.append(rating_human[key.lower()][key_aria])
                score_machine.append(rating_machine[key][key_aria])
            except KeyError:
                print(key, key_aria, 'not found')
    return np.array(score_human), np.array(score_machine)


def adjustHumanRatingJson(rating_human):

    rating_human_adjust = {}
    # adjust human rating dictionary
    for key in rating_human:
        dict_aria = {}
        for key_aria in rating_human[key]:
            if 'anchor' not in key_aria:  # remove anchor key
                value_aria = rating_human[key][key_aria]
                if isinstance(value_aria, list):  # remove reference key
                    id, _ = key_aria.split('_')
                    dict_aria[value_aria[1] + '_' + id] = value_aria[0]
        rating_human_adjust[key] = dict_aria
    return rating_human_adjust


def humanMachineCorrCoef(rating_human, rating_machine):

    score_human, score_machine = dict2score(rating_human, rating_machine)

    # r, p = pearsonr(score_human, score_SR_oracle)
    # print('pearson cof', r)
    r, p = spearmanr(score_human, score_machine)
    print('spearman cof', r)
    return r


if __name__ == '__main__':

    path_data = './data'

    rating_human = json.load(open(os.path.join(path_data, 'rating.json'), 'r'))
    rating_human_adjusted = adjustHumanRatingJson(rating_human)

    # rating_SR_oracle = json.load(open(os.path.join(path_data, 'rating_SR_oracle_total.json'), 'r'))
    # humanMachineCorrCoef(rating_human=rating_human_adjusted, rating_machine=rating_SR_oracle)

    mode = 'GOP'
    rating_oracle_total = json.load(open(os.path.join(path_data, 'rating_'+mode+'_oracle_total.json'), 'r'))
    rating_oracle_head = json.load(open(os.path.join(path_data, 'rating_'+mode+'_oracle_head.json'), 'r'))
    rating_oracle_belly = json.load(open(os.path.join(path_data, 'rating_'+mode+'_oracle_belly.json'), 'r'))
    humanMachineCorrCoef(rating_human=rating_human_adjusted, rating_machine=rating_oracle_total)
    humanMachineCorrCoef(rating_human=rating_human_adjusted, rating_machine=rating_oracle_head)
    humanMachineCorrCoef(rating_human=rating_human_adjusted, rating_machine=rating_oracle_belly)