import essentia.standard as es
import essentia
import pickle
import pandas as pd
# from pathName import *
import numpy
import os
from multiprocessing import Process
from src.utilFunctions import ensure_dir


def statsticsCal(array, dict_seg, desc):
    m = numpy.mean(array)
    v = numpy.var(array)
    d = numpy.diff(array)
    dm = numpy.mean(d)
    dv = numpy.var(d)
    dict_seg[desc + '.mean'] = m
    dict_seg[desc + '.var'] = v
    dict_seg[desc + '.dmean'] = dm
    dict_seg[desc + '.dvar'] = dv
    return  dict_seg


def frame_pool_aggregation(essentia_frame_pool, filename):
    """
    aggregate the feature pool into a pd dataframe
    :param essentia_frame_pool:
    :param filename:
    :return:
    """
    ii = 0
    feature_frame = pd.DataFrame()
    seg_framesize = 65  # hopsize = 1024, this will be 1.5s
    while ii < 432/seg_framesize+1:  # 432 is 10s
        dict_seg = {}

        # ignore all the useless features
        for desc in essentia_frame_pool.descriptorNames():

            if desc == 'lowlevel.gfcc.cov' or \
               desc == 'lowlevel.gfcc.icov' or \
               desc == 'lowlevel.mfcc.cov' or \
               desc == 'lowlevel.mfcc.icov':
                continue

            if 'onset_times' in desc or \
                'bpm_intervals' in desc or \
                'metadata' in desc or \
                'beats_position' in desc or \
                'chords_key' in desc or \
                'chords_scale' in desc or \
                'key_edma' in desc or \
                'key_krumhansl' in desc or \
                'key_temperley' in desc or \
                'chords_progression' in desc or \
                'rhythm' in desc or \
                'tonal.tuning_frequency' in desc or \
                'sfx.oddtoevenharmonicenergyratio' in desc or \
                'tristimulus' in desc or \
                'loudness_ebu128' in desc or \
                'histogram' in desc or \
                'melbands128' in desc:
                continue

            # if feature is a list, only use the first element
            feature = essentia_frame_pool[desc][0] if isinstance(essentia_frame_pool[desc], list) else essentia_frame_pool[desc]

            if type(feature) is float:
                continue

            if feature.shape[0] == 1:
                continue

            if len(feature.shape) == 1: # not frame-based feature
                dict_seg = statsticsCal(feature[seg_framesize*ii:seg_framesize*(ii+1)], dict_seg, desc)
            else:
                for jj in range(feature.shape[1]): # frame-based feature, calculate statistics
                    dict_seg = statsticsCal(feature[seg_framesize*ii:seg_framesize*(ii+1),jj], dict_seg, desc+str(jj))

        dataFrame_ii = pd.DataFrame(dict_seg, index=[filename + '_' + str(ii)])
        feature_frame = feature_frame.append(dataFrame_ii)
        ii += 1

    return feature_frame


def convert_essentia_pool_2_feature_array(essentia_pool):
    feature_array = []
    list_desc = []
    for desc in essentia_pool.descriptorNames():
        if "metadata" in desc or "histogram" in desc or type(essentia_pool[desc]) is not float:
            continue
        else:
            feature_array.append(essentia_pool[desc])
            list_desc.append(desc)
    return numpy.array(feature_array), list_desc


def subprocessFeatureExtractionFrame(path_root,
                                     path_folder,
                                     fn):
    """
    subprocess for essentia freesound feature extractor
    :param path_audio: input audio path
    :param fn: audio file name
    :return:
    """

    extractor = es.FreesoundExtractor(lowlevelFrameSize=2048, lowlevelHopSize=1024,
                                      tonalFrameSize=2048, tonalHopSize=1024,
                                      gfccStats=["mean"],
                                      mfccStats=["mean"],
                                      lowlevelStats=["mean", "stdev"],
                                      tonalStats=["mean", "stdev"],
                                      rhythmStats=["mean"])
    feature_aggregated, _ = extractor(os.path.join(path_root, path_folder, fn))
    feature_array, list_desc = convert_essentia_pool_2_feature_array(feature_aggregated)
    ensure_dir(os.path.join(path_root, path_folder+"_feature"))
    pickle.dump([feature_array, list_desc],
                open(os.path.join(path_root, path_folder+"_feature", fn+".pkl"), 'w'))


if __name__ == '__main__':

    root_phn_wav_path = "/Volumes/rong_segate/phoneme_audio_dlfm"
    sub_folders = ['student', 'extra_test']

    # old version of essentia processing can't free memory, use subprocess
    for folder in sub_folders:
        path_phn_wav = os.path.join(root_phn_wav_path, folder)

        filenames_audio = [f for f in os.listdir(path_phn_wav) if os.path.isfile(os.path.join(path_phn_wav, f))]
        for ii, fn in enumerate(filenames_audio):
            # if not os.path.isfile(os.path.join(path_feature_freesound_statistics_channel_string, fn.split('.')[0] + '.csv')):
            print('calculating', ii, fn, 'audio feature for', folder, 'in total', len(filenames_audio))
            p = Process(target=subprocessFeatureExtractionFrame,
                        args=(root_phn_wav_path,
                              folder,
                              fn,))
            p.start()
            p.join()
