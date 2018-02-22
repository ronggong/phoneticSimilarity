import matplotlib
matplotlib.use('TkAgg')

import json
import pickle
import numpy as np
from src.train_test_filenames import getTestRecordingsJoint
from src.filepath import *
from src.parameters import *
from src.phonemeMap import phns_tails

from src.textgridParser import textgridSyllablePhonemeParser
from src.audio_preprocessing import mfccDeltaDelta
from src.audio_preprocessing import segmentMfccLine
from src.utilFunctions import sterero2Mono
from src.distance_measures import BDDistanceMat

from baselineHelper import findShiftOffset
from baselineHelper import gaussianPipeline
from baselineHelper import removeNanRowCol
from baselineHelper import phnSequenceAlignment
from baselineHelper import getListsSylPhn
from baselineHelper import getIdxHeadsMissingTails


import soundfile as sf


def runProcess(val_test, mode):
    # the test dataset filenames
    primarySchool_val_recordings, primarySchool_test_recordings = getTestRecordingsJoint()

    if val_test == 'val':
        recordings = primarySchool_val_recordings
    else:
        recordings = primarySchool_test_recordings

    dict_distortion = {}
    dict_feature_phns = {}

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

        teacher_wav_file = os.path.join(primarySchool_wav_path, artist, 'teacher.wav')
        student_wav_file = os.path.join(primarySchool_wav_path, artist, fn + '.wav')

        print(teacher_wav_file)
        # calculate MFCCs
        # audio_data_teacher, fs_teacher = librosa.load(teacher_wav_file)
        # audio_data_student, fs_student = librosa.load(student_wav_file)

        audio_data_teacher, fs_teacher = sf.read(teacher_wav_file)
        audio_data_student, fs_student = sf.read(student_wav_file)

        audio_data_teacher = sterero2Mono(audio_data_teacher)
        audio_data_student = sterero2Mono(audio_data_student)

        # 39 dimensions mfcc
        mfccs_teacher = mfccDeltaDelta(audio_data=audio_data_teacher, fs=fs_teacher, framesize=framesize,
                                       hopsize=hopsize)
        mfccs_student = mfccDeltaDelta(audio_data=audio_data_student, fs=fs_student, framesize=framesize,
                                       hopsize=hopsize)

        # create the artist key
        if artist not in dict_distortion:
            dict_distortion[artist] = {}
            dict_feature_phns[artist] = {}

        for ii_line in range(len(studentPhonemeLists)):  # iterate each line

            # find the right line index for the teacher's textgrid,
            # ``student02_first_half'' only corresponds to a part of the teacher's textgrid,
            # we need to shift the index of the teacher's textgrid to find the right line
            ii_aug = findShiftOffset(gtSyllableLists=studentSyllableLists,
                                     scoreSyllableLists=teacherSyllableLists,
                                     ii_line=ii_line)

            # trim the mfccs line
            line_teacher = teacherPhonemeLists[ii_line + ii_aug][0]
            mfccs_teacher_line = segmentMfccLine(line=line_teacher, hopsize_t=hopsize_t, mfccs=mfccs_teacher)

            line_student = studentPhonemeLists[ii_line][0]
            mfccs_student_line = segmentMfccLine(line=line_student, hopsize_t=hopsize_t, mfccs=mfccs_student)

            list_phn_teacher, list_phn_student, list_syl_teacher, list_syl_onsets_time_teacher = \
                getListsSylPhn(teacherSyllableLists=teacherSyllableLists,
                               teacherPhonemeLists=teacherPhonemeLists,
                               studentPhonemeLists=studentPhonemeLists,
                               ii_line=ii_line,
                               ii_aug=ii_aug)

            phns_teacher = [lpt[2] for lpt in list_phn_teacher]
            phns_student = [lpt[2] for lpt in list_phn_student]

            insertion_indices_student, deletion_indices_teacher, teacher_student_indices_pair, _ = \
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

            mu_cov_teacher = []
            mu_cov_student = []

            for ii_phn_pair in range(len(list_phn_teacher_pair)):
                mu_teacher, cov_teacher, phn_label_teacher = gaussianPipeline(list_phn=list_phn_teacher_pair,
                                                                              ii=ii_phn_pair,
                                                                              hopsize_t=hopsize_t,
                                                                              mfccs_line=mfccs_teacher_line)
                mu_cov_teacher.append([mu_teacher, cov_teacher])

                mu_student, cov_student, phn_label_student = gaussianPipeline(list_phn=list_phn_student_pair,
                                                                              ii=ii_phn_pair,
                                                                              hopsize_t=hopsize_t,
                                                                              mfccs_line=mfccs_student_line)
                mu_cov_student.append([mu_student, cov_student])

            distance_mat_teacher = BDDistanceMat(mu_cov_teacher)
            distance_mat_student = BDDistanceMat(mu_cov_student)

            if mode == '_total':
                # remove the matrix row and col if containing nan
                distance_mat_teacher, distance_mat_student, _ = removeNanRowCol(distance_mat_teacher,
                                                                                distance_mat_student)


                # print((distance_mat_teacher - distance_mat_student).shape, distance_mat_teacher.shape[0])

                distortion = np.linalg.norm(distance_mat_teacher - distance_mat_student) / distance_mat_teacher.shape[0]

                # phone-level distortion
                distortion_phns = np.linalg.norm(distance_mat_teacher - distance_mat_student, axis=1) / np.sqrt(
                    distance_mat_teacher.shape[0])


            elif mode == '_head':  # only consider head phns
                print(idx_syl_heads)
                distance_head_mat_teacher = distance_mat_teacher[np.ix_(idx_syl_heads, idx_syl_heads)]
                distance_head_mat_student = distance_mat_student[np.ix_(idx_syl_heads, idx_syl_heads)]

                distance_head_mat_teacher, distance_head_mat_student, _ = \
                    removeNanRowCol(distance_head_mat_teacher, distance_head_mat_student)

                # print((distance_head_mat_teacher-distance_head_mat_student).shape, distance_head_mat_teacher.shape[0])

                distortion = np.linalg.norm(distance_head_mat_teacher - distance_head_mat_student) / distance_head_mat_teacher.shape[0]

                # phone-level distortion
                distortion_phns = np.linalg.norm(distance_head_mat_teacher - distance_head_mat_student, axis=1) / np.sqrt(
                    distance_head_mat_teacher.shape[0])

            elif mode == '_belly':
                idx_syl_belly = [ii_entire_idx for ii_entire_idx in range(distance_mat_teacher.shape[0]) if
                                 ii_entire_idx not in idx_syl_heads]

                distance_belly_mat_teacher = distance_mat_teacher[np.ix_(idx_syl_belly, idx_syl_belly)]
                distance_belly_mat_student = distance_mat_student[np.ix_(idx_syl_belly, idx_syl_belly)]

                distance_belly_mat_teacher, distance_belly_mat_student, _ = \
                    removeNanRowCol(distance_belly_mat_teacher, distance_belly_mat_student)

                distortion = np.linalg.norm(distance_belly_mat_teacher - distance_belly_mat_student) / distance_belly_mat_teacher.shape[0]

                # phone-level distortion
                distortion_phns = np.linalg.norm(distance_belly_mat_teacher - distance_belly_mat_student, axis=1) / np.sqrt(
                    distance_belly_mat_teacher.shape[0])

            if np.isnan(distortion):
                raise ValueError

            dict_distortion[artist][fn + '_' + str(ii_line + ii_aug)] = distortion
            dict_feature_phns[artist][fn + '_' + str(ii_line + ii_aug)] = {'distortion_phns':distortion_phns, 'num_tails_missing':num_tails_missing}

    if val_test == 'test':
        with open('./data/rating_SR_oracle' + mode + '.json', 'w') as savefile:
            json.dump(dict_distortion, savefile)

    with open('./data/training_features/SR_oracle_'+ val_test + mode + '.pkl', 'wb') as savefile:
        pickle.dump(dict_feature_phns, savefile)


if __name__ == '__main__':

    val_test = 'test'

    for mode in ['_total' , '_head', '_belly']: # head, belly, linearReg, linearReg_missing_tails

        runProcess(val_test=val_test, mode=mode)