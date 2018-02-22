from zhon.hanzi import punctuation as puncChinese
import string
import re
import nwalign3 as nw
import numpy as np
from sklearn.mixture import GaussianMixture
from src.phonemeMap import pho_set_initials


def removePunctuation(char):
    if len(re.findall(r'[\u4e00-\u9fff]+', char)):
        char = re.sub(r"[%s]+" % puncChinese, "", char)
    return char


def findShiftOffset(gtSyllableLists, scoreSyllableLists, ii_line):
    """
    find the shifting Offset
    :param gtSyllableLists:
    :param scoreSyllableLists:
    :param ii_line:
    :return:
    """
    ii_aug = 0
    text_gt = removePunctuation(gtSyllableLists[ii_line][0][2].rstrip())
    text_score = removePunctuation(scoreSyllableLists[ii_line + ii_aug][0][2].rstrip())

    while text_gt != text_score:
        ii_aug += 1
        text_score = removePunctuation(scoreSyllableLists[ii_line + ii_aug][0][2].rstrip())
    return ii_aug


def convertSyl2Letters(syllables0, syllables1):
    """
    convert syllable lists to letter string
    :param syllables0:
    :param syllables1:
    :return:
    """
    dict_letters2syl = {}
    dict_syl2letters = {}
    ascii_letters = string.ascii_letters
    for ii, syl in enumerate(list(set(syllables0+syllables1))):
        dict_letters2syl[ascii_letters[ii]] = syl
        dict_syl2letters[syl] = ascii_letters[ii]

    syllables0_converted = ''.join([dict_syl2letters[syl] for syl in syllables0])
    syllables1_converted = ''.join([dict_syl2letters[syl] for syl in syllables1])
    return syllables0_converted, syllables1_converted, dict_letters2syl


def identifyInsertionDeletionIdx(phns_teacher_letters, phns_student_letters, dict_letters2syl):
    """
    identify which index in teacher's phone list is a deletion
    which index in student's phone list is a insertion
    the dictionary map student phone list index to teacher's phone

    Example:
    teacher phone list cd-cdc
    student phone list cdlc-c

    The 3rd phone of student is a insertion, the 4th phone of teacher is a deletion
    the corresponding dictionary is {0: c, 1, d, 3: c, 4:c}

    :param phns_teacher_letters:
    :param phns_student_letters:
    :param dict_letters2syl:
    :return:
    """
    ii_teacher, ii_student = 0, 0
    insertion_indices, deletion_indices = [], []
    dict_student_idx_2_teacher_phn = {}
    teacher_student_indices_pair = [] # the indices corresponded between teacher and student phone lists

    for ii in range(len(phns_teacher_letters)):
        if phns_teacher_letters[ii] != '-' and phns_student_letters[ii] != '-':
            dict_student_idx_2_teacher_phn[ii_student] = dict_letters2syl[phns_teacher_letters[ii]]
            teacher_student_indices_pair.append([ii_teacher, ii_student])
            ii_teacher += 1
            ii_student += 1
        else:
            if phns_teacher_letters[ii] == '-':
                insertion_indices.append(ii_student)
            elif phns_student_letters[ii] == '-':
                deletion_indices.append(ii_teacher)

            if phns_teacher_letters[ii] != '-':
                ii_teacher += 1
            if phns_student_letters[ii] != '-':
                ii_student += 1

    return dict_student_idx_2_teacher_phn, insertion_indices, deletion_indices, teacher_student_indices_pair


def removeSilence(list_phns):
    """
    Remove silence phoneme onsets and labels
    :param phn_onsets:
    :param phn_labels:
    :return:
    """
    for ii in reversed(range(len(list_phns))):
        if list_phns[ii][2] == u'':
            list_phns.pop(ii)
    return list_phns


def gaussianPipeline(list_phn, ii, hopsize_t, mfccs_line):
    """
    pipeline to fix the gaussian for a mfccs segment
    :param list_phn:
    :param ii:
    :param hopsize_t:
    :param mfccs_line: (feature, frames)
    :return:
    """
    phn_start_frame = \
        int(round((list_phn[ii][0] - list_phn[0][0]) / hopsize_t))
    phn_end_frame = \
        int(round((list_phn[ii][1] - list_phn[0][0]) / hopsize_t))
    phn_label = list_phn[ii][2]

    if phn_end_frame == phn_start_frame: # if no frame in this segment, just put 0 for mu and cov
        mu = np.nan
        cov = np.nan
    else:
        mfccs_phn = mfccs_line[:, phn_start_frame:phn_end_frame]
        gmm = GaussianMixture(n_components=1, covariance_type='diag')
        gmm.fit(mfccs_phn.T)
        mu = gmm.means_[0,:]
        cov = gmm.covariances_[0,:]

    return mu, cov, phn_label


def removeNanRowCol(distance_mat_teacher, distance_mat_student):
    distance_mat_teacher_nan = distance_mat_teacher.copy()
    distance_mat_student_nan = distance_mat_student.copy()
    distance_mat_teacher_nan[distance_mat_teacher_nan == 0] = np.nan
    distance_mat_student_nan[distance_mat_student_nan == 0] = np.nan

    idx_nan_teacher = np.argwhere(np.isnan(distance_mat_teacher_nan).all(axis=1))
    idx_nan_student = np.argwhere(np.isnan(distance_mat_student_nan).all(axis=1))

    idx_nan = list(set(list(idx_nan_teacher.flatten()) + list(idx_nan_student.flatten())))
    if len(idx_nan):
        distance_mat_teacher = np.delete(distance_mat_teacher, idx_nan, axis=0)
        distance_mat_student = np.delete(distance_mat_student, idx_nan, axis=0)
        distance_mat_teacher = np.delete(distance_mat_teacher, idx_nan, axis=1)
        distance_mat_student = np.delete(distance_mat_student, idx_nan, axis=1)

    return distance_mat_teacher, distance_mat_student, idx_nan


def phnSequenceAlignment(phns_teacher, phns_student):
    """
    Align two phn sequences
    :param phns_teacher:
    :param phn_student:
    :return:
    """
    # convert phonemes to letters
    phns_teacher_letters, phns_student_letters, dict_letters2syl = \
        convertSyl2Letters(syllables0=phns_teacher, syllables1=phns_student)

    # global alignment, because the mismatch between teacher and student phone list
    phns_teacher_aligned, phns_student_aligned = \
        nw.global_align(phns_teacher_letters, phns_student_letters)

    # output the insertion and deletion indices, and the corresponding teacher's phones for the student' phones
    dict_student_idx_2_teacher_phn, insertion_indices_student, deletion_indices_teacher, teacher_student_indices_pair = \
        identifyInsertionDeletionIdx(phns_teacher_aligned, phns_student_aligned, dict_letters2syl)

    return insertion_indices_student, deletion_indices_teacher, teacher_student_indices_pair, dict_student_idx_2_teacher_phn


def getListsSylPhn(teacherSyllableLists, teacherPhonemeLists, studentPhonemeLists, ii_line, ii_aug):
    """
    like the function name
    :param teacherSyllableLists:
    :param teacherPhonemeLists:
    :param studentPhoneme:
    :param ii_line:
    :param ii_aug:
    :return:
    """
    list_syl_teacher = teacherSyllableLists[ii_line + ii_aug][1]

    # syllable onset time
    list_syl_onsets_time_teacher = [s[0] for s in list_syl_teacher]

    list_phn_teacher = teacherPhonemeLists[ii_line + ii_aug][1]
    list_phn_student = studentPhonemeLists[ii_line][1]

    # remove the silence phone from the list
    list_phn_teacher = removeSilence(list_phn_teacher)
    list_phn_student = removeSilence(list_phn_student)

    return list_phn_teacher, list_phn_student, list_syl_teacher, list_syl_onsets_time_teacher


def getIdxHeadsMissingTails(teacher_student_indices_pair,
                            list_phn_teacher,
                            list_phn_student,
                            list_syl_onsets_time_teacher,
                            deletion_indices_teacher,
                            phns_tails):
    # extract the paired phones between teacher and student
    list_phn_teacher_pair, list_phn_student_pair = [], []
    for indices_pair in teacher_student_indices_pair:
        list_phn_teacher_pair.append(list_phn_teacher[indices_pair[0]])
        list_phn_student_pair.append(list_phn_student[indices_pair[1]])

    # get syllable head index
    idx_syl_heads = [ii_lptp for ii_lptp, lptp in enumerate(list_phn_teacher_pair) if
                     lptp[0] in list_syl_onsets_time_teacher and lptp[2] in pho_set_initials]

    # count missing phn tails
    phn_tails_missing = [list_phn_teacher[dit][2] for dit in deletion_indices_teacher if
                         list_phn_teacher[dit][2] in phns_tails]
    num_tails_missing = len(phn_tails_missing)

    return list_phn_teacher_pair, list_phn_student_pair, idx_syl_heads, phn_tails_missing, num_tails_missing

if __name__ == '__main__':
    # test
    dict_student_idx_2_teacher_phn, insertion_indices, deletion_indices, _ = \
    identifyInsertionDeletionIdx(phns_teacher_letters='cd-cdclcc-d',
                                 phns_student_letters='cdlc-c-ccdd',
                                 dict_letters2syl={'c':'c', 'd':'d', 'l':'l'})

    print(dict_student_idx_2_teacher_phn)
    print(insertion_indices)
    print(deletion_indices)

    print(removeSilence([[0, 1, 'a'], [2, 3, ''], [4, 5, 'b']]))