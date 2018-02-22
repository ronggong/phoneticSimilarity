"""
parse the human rating to a json
"""
import matplotlib
matplotlib.use('TkAgg')

from src.filepath import *
from yaml import load
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt


def parseYaml(artist_aria):
    """
    parse yaml file into a dictionary
    :param filename_ymal:
    :return:
    """
    filename_ymal = os.path.join(path_ymal_config, artist_aria[1]+'.yaml')

    with open(filename_ymal, 'r') as stream:
        yaml_config = load(stream)

    # parse yaml to a dictionary
    dict = {}
    for list_yaml in yaml_config['pages']:
        if list_yaml['content'] == 'please rate each singing sample':
            name = list_yaml['name']
            id = list_yaml['id']
            stimuli = list_yaml['stimuli']
            for key in stimuli:
                filename_student = stimuli[key].split('/')[-1].split('.')[0].replace(id+'_', '', 1)
                dict[id+'_'+key] = filename_student
    return dict


def parseMushraCsv(artist_aria):
    """
    parse Mushra csv into a dictionary
    :param artist_aria:
    :return:
    """
    dict = {}
    filename_mushra_csv = os.path.join(path_rating, artist_aria[0] + '-' + artist_aria[1], 'mushra.csv')
    dataframe = pd.read_csv(filename_mushra_csv)
    dataframe_stimuli = dataframe[dataframe['trial_id'].str.contains('training_page') == False]
    for id, stimuli, score in zip(dataframe_stimuli['trial_id'].values, dataframe_stimuli['rating_stimulus'],
                                  dataframe_stimuli['rating_score']):
        dict[id+'_'+stimuli] = float(score)
    return dict

if __name__ == '__main__':
    path_ymal_config = os.path.join(primarySchool_human_rating_path, 'pronunciation_config')
    path_rating = os.path.join(primarySchool_human_rating_path, 'pronunciation')

    # human rating
    list_artist_aria = []
    directories_rating = [x[0] for x in os.walk(path_rating)][1:]
    for directory in directories_rating:
        artist_aria = directory.split('/')[-1]
        artist = artist_aria.split('-')[0]
        aria = artist_aria.replace(artist+'-', '')
        list_artist_aria.append([artist, aria])

    reference_score = [] # score for all references
    anchor_diff = [] # difference score for the anchor
    rating = {}
    line_num = 0
    all_rating = []
    for artist_aria in list_artist_aria:
        dict_mushra = parseMushraCsv(artist_aria)
        dict_ymal = parseYaml(artist_aria)

        dict_all = dict_mushra.copy()

        for key in dict_ymal:
            dict_all[key] = [dict_mushra[key], dict_ymal[key]]
            line_num += 1
        rating[artist_aria[0]+'/'+artist_aria[1]] = dict_all

        for key in dict_all:
            if 'reference' in key:
                reference_score.append(dict_all[key])
            if 'anchor' in key:
                anchor_diff.append(np.abs(dict_all[key][0] - dict_all[key.replace('_anchor', '')][0]))
            if 'reference' not in key and 'anchor' not in key:
                all_rating.append(dict_all[key][0])

    print(line_num)
    print(np.mean(all_rating), np.std(all_rating))
    # rater validity test
    print(np.mean(reference_score))
    print(np.mean(anchor_diff))

    plt.figure()
    plt.hist(all_rating, bins=50)
    plt.show()

    with open('./data/rating.json', 'w') as rating_file:
        json.dump(rating, rating_file)