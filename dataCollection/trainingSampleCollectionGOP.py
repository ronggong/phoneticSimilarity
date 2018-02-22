import numpy as np
from src.filePathGOPmodel import *
from src.parameters import *
from src.phonemeMap import dic_pho_map, dic_pho_label
from src.textgridParser import syllableTextgridExtraction
from src.audio_preprocessing import getMFCCBands2DMadmom
from src.audio_preprocessing import featureReshape
from sklearn import preprocessing


def dumpFeaturePho(wav_path,
                   textgrid_path,
                   recordings,
                   syllableTierName,
                   phonemeTierName):
    '''
    dump the MFCC for each phoneme
    :param recordings:
    :return:
    '''

    ##-- dictionary feature
    dic_pho_feature = {}

    for _,pho in enumerate(set(dic_pho_map.values())):
        dic_pho_feature[pho] = np.array([])

    for artist_path, recording in recordings:
        nestedPhonemeLists, numSyllables, numPhonemes   \
            = syllableTextgridExtraction(textgrid_path,
                                         join(artist_path,recording),
                                         syllableTierName,
                                         phonemeTierName)

        # audio
        wav_full_filename   = join(wav_path,artist_path,recording+'.wav')

        mfcc = getMFCCBands2DMadmom(wav_full_filename, fs, hopsize_t, channel=1)

        for ii,pho in enumerate(nestedPhonemeLists):
            print('calculating ', recording, ' and phoneme ', str(ii), ' of ', str(len(nestedPhonemeLists)))
            for p in pho[1]:
                # map from annotated xsampa to readable notation
                try:
                    key = dic_pho_map[p[2]]
                except KeyError:
                    print(artist_path, recording)
                    print(ii, p[2])
                    raise

                sf = int(round(p[0] * fs / float(hopsize))) # starting frame
                ef = int(round(p[1] * fs / float(hopsize))) # ending frame

                mfcc_p = mfcc[sf:ef,:]  # phoneme syllable

                if not len(dic_pho_feature[key]):
                    dic_pho_feature[key] = mfcc_p
                else:
                    dic_pho_feature[key] = np.vstack((dic_pho_feature[key],mfcc_p))

    return dic_pho_feature

def featureAggregator(dic_pho_feature_train):
    """
    aggregate feature dictionary into numpy feature, label lists,
    reshape the feature
    :param dic_pho_feature_train:
    :return:
    """
    feature_all = np.array([], dtype='float32')
    label_all = []
    for key in dic_pho_feature_train:
        feature = dic_pho_feature_train[key]
        label = [dic_pho_label[key]] * len(feature)

        if len(feature):
            if not len(feature_all):
                feature_all = feature
            else:
                feature_all = np.vstack((feature_all, feature))
            label_all += label
    label_all = np.array(label_all, dtype='int64')

    scaler = preprocessing.StandardScaler().fit(feature_all)
    feature_all = scaler.transform(feature_all)
    feature_all = featureReshape(feature_all, nlen=7)

    return feature_all, label_all, scaler


if __name__ == '__main__':

    from src.train_test_filenames import getTeacherRecordings
    import h5py
    import pickle

    trainNacta2017, trainNacta, trainSepa, trainPrimarySchool = getTeacherRecordings()

    dic_pho_feature_nacta2017 = dumpFeaturePho(wav_path=nacta2017_wav_path,
                                               textgrid_path=nacta2017_textgrid_path,
                                               recordings=trainNacta2017,
                                               syllableTierName='line',
                                               phonemeTierName='details')

    dic_pho_feature_nacta = dumpFeaturePho(wav_path=nacta_wav_path,
                                               textgrid_path=nacta_textgrid_path,
                                               recordings=trainNacta,
                                               syllableTierName='line',
                                               phonemeTierName='details')

    dic_pho_feature_primarySchool = dumpFeaturePho(wav_path=primarySchool_wav_path,
                                                   textgrid_path=primarySchool_textgrid_path,
                                                   recordings=trainPrimarySchool,
                                                   syllableTierName='line',
                                                   phonemeTierName='details')

    dic_pho_feature_sepa = dumpFeaturePho(wav_path=nacta_wav_path,
                                           textgrid_path=nacta_textgrid_path,
                                           recordings=trainSepa,
                                           syllableTierName='line',
                                           phonemeTierName='details')

    # fuse two dictionaries
    list_key = list(set(list(dic_pho_feature_nacta.keys()) + list(dic_pho_feature_nacta2017.keys()) +
                        list(dic_pho_feature_primarySchool.keys()) + list(dic_pho_feature_sepa.keys())))
    print(list_key)

    dic_pho_feature_all = {}
    for key in list_key:
        if not len(dic_pho_feature_nacta2017[key]):
            dic_pho_feature_nacta2017[key] = np.empty((0, 1200), dtype='float32')

        if not len(dic_pho_feature_nacta[key]):
            dic_pho_feature_nacta[key] = np.empty((0, 1200), dtype='float32')

        if not len(dic_pho_feature_primarySchool[key]):
            dic_pho_feature_primarySchool[key] = np.empty((0, 1200), dtype='float32')

        if not len(dic_pho_feature_sepa[key]):
            dic_pho_feature_sepa[key] = np.empty((0, 1200), dtype='float32')

        dic_pho_feature_all[key] = np.vstack((dic_pho_feature_nacta[key], dic_pho_feature_nacta2017[key],
                                              dic_pho_feature_primarySchool[key], dic_pho_feature_sepa[key]))


    feature_all, label_all, scaler = featureAggregator(dic_pho_feature_all)

    print(feature_all.shape)
    print(label_all.shape)

    # save feature label scaler
    filename_feature_all = join(data_path_GOP_model, 'feature_gop.h5')
    h5f = h5py.File(filename_feature_all, 'w')
    h5f.create_dataset('feature_all', data=feature_all)
    h5f.close()

    pickle.dump(label_all,
                 open(join(data_path_GOP_model, 'labels_gop.pkl'),'wb'), protocol=2)

    pickle.dump(scaler,
                open(join(data_path_GOP_model, 'scaler_gop.pkl'), 'wb'), protocol=2)