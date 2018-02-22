import numpy as np
from scipy.spatial.distance import cosine
from src.phonemeMap import dic_pho_map
from src.phonemeMap import dic_pho_label
from src.audio_preprocessing import featureReshape


def GOP_phn_level(phn_label, obs_line_phn):
    """
    Goodness of pronunciation phoneme level
    :param phn_label:
    :param obs_line_phn:
    :return:
    """
    # calculate GOP
    idx_phn = dic_pho_label[dic_pho_map[phn_label]]
    numerator = np.sum(obs_line_phn[:, idx_phn])
    denumerator = np.sum(np.max(obs_line_phn, axis=1))
    # print('n', numerator)
    # print('d', denumerator)
    GOP_phn = np.abs((numerator - denumerator)) / obs_line_phn.shape[0]
    return GOP_phn


def measureEmbDissimilarity(model_keras_cnn_0, log_mel_phn_teacher, log_mel_phn_student):
    """obtain the embedding dissimilarity"""
    log_mel_phn_teacher = np.expand_dims(log_mel_phn_teacher, axis=0)
    emb_phn_teacher = model_keras_cnn_0.predict_on_batch(log_mel_phn_teacher)
    log_mel_phn_student = np.expand_dims(log_mel_phn_student, axis=0)
    emb_phn_student = model_keras_cnn_0.predict_on_batch(log_mel_phn_student)
    dis_dis = 1.0 - cosine(emb_phn_teacher, emb_phn_student)
    return dis_dis

def measureEmbFrameLevelDissimilarity(model_keras_cnn_0, log_mel_phn_teacher, log_mel_phn_student):
    """obtain the frame level embedding dissimilarity"""
    log_mel_phn_teacher = np.expand_dims(featureReshape(log_mel_phn_teacher, nlen=7), axis=1)
    emb_phn_teacher = model_keras_cnn_0.predict_on_batch(log_mel_phn_teacher)
    emb_phn_teacher = np.mean(emb_phn_teacher, axis=0)

    log_mel_phn_student = np.expand_dims(featureReshape(log_mel_phn_student, nlen=7), axis=1)
    emb_phn_student = model_keras_cnn_0.predict_on_batch(log_mel_phn_student)
    emb_phn_student = np.mean(emb_phn_student, axis=0)

    # print(emb_phn_teacher)
    # print(emb_phn_student)
    dis_dis = 1.0 - cosine(emb_phn_teacher, emb_phn_student)
    return dis_dis