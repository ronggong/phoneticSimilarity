import matplotlib
matplotlib.use('TkAgg')

import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from src.train_test_filenames import getTestRecordingsJoint
from src.filepath import *
from src.parameters import *
from src.phonemeMap import phns_tails

from src.textgridParser import textgridSyllablePhonemeParser
from src.audio_preprocessing import getMFCCBands2DMadmom
from src.audio_preprocessing import featureReshape

from baselineHelper import findShiftOffset
from baselineHelper import getListsSylPhn
from baselineHelper import phnSequenceAlignment
from baselineHelper import getIdxHeadsMissingTails

from phonetic_assessment import GOP_phn_level

from keras.models import load_model

def figurePlot(obs):
    # plot Error analysis figures
    plt.figure()
    plt.imshow(obs)
    plt.xlabel('phone indices', fontsize=12)
    plt.xlabel('frames', fontsize=12)
    plt.axis('tight')
    plt.show()

def disLinePlot(dis, list_phn):
    """plot the dissimilarity list line"""
    plt.figure()
    plt.bar(np.arange(len(dis)), dis, alpha=0.7, align='center')
    plt.xticks(np.arange(len(list_phn)), list_phn)
    plt.show()

def getObsLine(studentPhonemeLists, ii_line, hopsize_t, log_mel_reshaped, model_keras_cnn_0):
    line = studentPhonemeLists[ii_line][0]
    # start and end time
    time_start = line[0]
    time_end = line[1]
    frame_start = int(round(time_start / hopsize_t))
    frame_end = int(round(time_end / hopsize_t))

    # log_mel_reshape line
    log_mel_reshaped_line = log_mel_reshaped[frame_start: frame_end]
    log_mel_reshaped_line = np.expand_dims(log_mel_reshaped_line, axis=1)

    # emission probabilities
    obs_line = model_keras_cnn_0.predict(log_mel_reshaped_line, batch_size=128, verbose=0)
    obs_line = np.log(obs_line+1e-128)
    return obs_line


def runProcess(val_test, plot):
    model_keras_cnn_0 = load_model(kerasModels_path)

    # open a pickle from python 2 in python 3, requires to add encoding
    scaler = pickle.load(open(kerasScaler_path, 'rb'), encoding='latin1')

    # the test dataset filenames
    primarySchool_val_recordings, primarySchool_test_recordings = getTestRecordingsJoint()

    if val_test == 'val':
        recordings = primarySchool_val_recordings
    else:
        recordings = primarySchool_test_recordings

    dict_total = {}
    dict_head = {}
    dict_belly = {}

    dict_feature_phns_total = {}
    dict_feature_phns_head = {}
    dict_feature_phns_belly = {}

    for artist, fn in recordings:

        # teacher's textgrid file
        teacher_textgrid_file = os.path.join(primarySchool_textgrid_path, artist, 'teacher.TextGrid')
        # textgrid path, to get the line onset offset
        student_textgrid_file = os.path.join(primarySchool_textgrid_path, artist, fn + '.TextGrid')

        # parse the textgrid to phoneme list
        teacherSyllableLists, teacherPhonemeLists = textgridSyllablePhonemeParser(teacher_textgrid_file,
                                                                              'dianSilence',
                                                                              'details')
        studentSyllableLists, studentPhonemeLists = textgridSyllablePhonemeParser(student_textgrid_file,
                                                                        'dianSilence',
                                                                        'details')

        student_wav_file = os.path.join(primarySchool_wav_path, artist, fn + '.wav')

        # calculate log mel
        log_mel = getMFCCBands2DMadmom(student_wav_file, fs, hopsize_t, channel=1)
        log_mel_scaled = scaler.transform(log_mel)
        log_mel_reshaped = featureReshape(log_mel_scaled, nlen=7)

        if artist not in dict_total:
            dict_total[artist] = {}
            dict_head[artist] = {}
            dict_belly[artist] = {}

            dict_feature_phns_total[artist] = {}
            dict_feature_phns_head[artist] = {}
            dict_feature_phns_belly[artist] = {}

        for ii_line in range(len(studentPhonemeLists)): # iterate each line

            # find the right line index for the teacher's textgrid,
            # ``student02_first_half'' only corresponds to a part of the teacher's textgrid,
            # we need to shift the index of the teacher's textgrid to find the right line
            ii_aug = findShiftOffset(gtSyllableLists=studentSyllableLists,
                                     scoreSyllableLists=teacherSyllableLists,
                                     ii_line=ii_line)

            list_phn_teacher, list_phn_student, list_syl_teacher, list_syl_onsets_time_teacher = \
                getListsSylPhn(teacherSyllableLists=teacherSyllableLists,
                               teacherPhonemeLists=teacherPhonemeLists,
                               studentPhonemeLists=studentPhonemeLists,
                               ii_line=ii_line,
                               ii_aug=ii_aug)


            phns_teacher = [lpt[2] for lpt in list_phn_teacher]
            phns_student = [lpt[2] for lpt in list_phn_student]

            insertion_indices_student, deletion_indices_teacher, teacher_student_indices_pair, dict_student_idx_2_teacher_phn = \
                phnSequenceAlignment(phns_teacher=phns_teacher, phns_student=phns_student)

            list_phn_teacher_pair, list_phn_student_pair, idx_syl_heads, phn_tails_missing, num_tails_missing = \
                getIdxHeadsMissingTails(teacher_student_indices_pair=teacher_student_indices_pair,
                                        list_phn_teacher=list_phn_teacher,
                                        list_phn_student=list_phn_student,
                                        list_syl_onsets_time_teacher=list_syl_onsets_time_teacher,
                                        deletion_indices_teacher=deletion_indices_teacher,
                                        phns_tails=phns_tails)

            print('these phone indices are inserted in student phone list', insertion_indices_student)
            print('these phone indices are deleted in teacher phone list', deletion_indices_teacher)
            print('these phone tails are deleted in teacher phone list', phn_tails_missing)

            obs_line = getObsLine(studentPhonemeLists=studentPhonemeLists,
                                   ii_line=ii_line,
                                   hopsize_t=hopsize_t,
                                   log_mel_reshaped=log_mel_reshaped,
                                   model_keras_cnn_0=model_keras_cnn_0)

            GOP_line = []
            for ii_phn in range(len(list_phn_student_pair)):

                phn_start_frame = int(round((list_phn_student_pair[ii_phn][0] - list_phn_student_pair[0][0]) / hopsize_t))
                phn_end_frame = int(round((list_phn_student_pair[ii_phn][1] - list_phn_student_pair[0][0]) / hopsize_t))

                phn_label = list_phn_teacher_pair[ii_phn][2]

                # the case of the phn length is 0
                if phn_end_frame == phn_start_frame:
                    GOP_line.append([ii_phn, -np.inf, phn_label])
                    continue

                obs_line_phn = obs_line[phn_start_frame:phn_end_frame]

                # if plot:
                #     figurePlot(obs_line_phn.T)

                # calculate GOP
                GOP_phn = GOP_phn_level(phn_label=phn_label, obs_line_phn=obs_line_phn)
                GOP_line.append([ii_phn, GOP_phn, phn_label])

            # print(len(GOP_line), idx_syl_heads)
            gop_total = [gop[1] for gop in GOP_line if not np.isinf(gop[1])]
            gop_head = [gop[1] for gop in GOP_line if not np.isinf(gop[1]) and gop[0] in idx_syl_heads]
            gop_belly = [gop[1] for gop in GOP_line if not np.isinf(gop[1]) and gop[0] not in idx_syl_heads]

            if plot:
                disLinePlot(gop_total, [gop[2] for gop in GOP_line if not np.isinf(gop[1])])
                disLinePlot(gop_head, [gop[2] for gop in GOP_line if not np.isinf(gop[1]) and gop[0] in idx_syl_heads])
                disLinePlot(gop_belly, [gop[2] for gop in GOP_line if not np.isinf(gop[1]) and gop[0] not in idx_syl_heads])


            total_distortion = np.mean(gop_total)
            head_distortion = np.mean(gop_head)
            belly_distortion = np.mean(gop_belly)

            dict_total[artist][fn + '_' + str(ii_line+ii_aug)] = total_distortion
            dict_head[artist][fn + '_' + str(ii_line+ii_aug)] = head_distortion
            dict_belly[artist][fn + '_' + str(ii_line+ii_aug)] = belly_distortion

            dict_feature_phns_total[artist][fn + '_' + str(ii_line+ii_aug)] = {'distortion_phns':np.array(gop_total), 'num_tails_missing':num_tails_missing}
            dict_feature_phns_head[artist][fn + '_' + str(ii_line+ii_aug)] = {'distortion_phns':np.array(gop_head), 'num_tails_missing':num_tails_missing}
            dict_feature_phns_belly[artist][fn + '_' + str(ii_line+ii_aug)] = {'distortion_phns':np.array(gop_belly), 'num_tails_missing':num_tails_missing}

    if val_test == 'test':
        with open('./data/rating_GOP_oracle_total.json', 'w') as savefile:
            json.dump(dict_total, savefile)
        with open('./data/rating_GOP_oracle_head.json', 'w') as savefile:
            json.dump(dict_head, savefile)
        with open('./data/rating_GOP_oracle_belly.json', 'w') as savefile:
            json.dump(dict_belly, savefile)

    with open('./data/training_features/GOP_oracle_'+val_test+'_total.pkl', 'wb') as savefile:
        pickle.dump(dict_feature_phns_total, savefile)
    with open('./data/training_features/GOP_oracle_'+val_test+'_head.pkl', 'wb') as savefile:
        pickle.dump(dict_feature_phns_head, savefile)
    with open('./data/training_features/GOP_oracle_'+val_test+'_belly.pkl', 'wb') as savefile:
        pickle.dump(dict_feature_phns_belly, savefile)

if __name__ == '__main__':

    plot = False

    for val_test in ['val', 'test']:

        runProcess(val_test=val_test, plot=plot)
