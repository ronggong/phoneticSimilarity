import matplotlib
matplotlib.use('TkAgg')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from src.train_test_filenames import getTestRecordingsJoint
from src.filepath import *
from src.parameters import *
from src.phonemeMap import pho_consonants

from src.textgridParser import textGrid2WordList
from src.audio_preprocessing import getMFCCBands2DMadmom
from src.audio_preprocessing import featureReshape
from src.audio_preprocessing import VAD

from phonetic_assessment import GOP_phn_level

from keras.models import load_model

def figurePlot(mfcc_line, vad_line):
    # plot Error analysis figures
    plt.figure(figsize=(16, 4))
    # plt.figure(figsize=(8, 4))
    # class weight
    ax1 = plt.subplot(211)
    y = np.arange(0, 80)
    x = np.arange(0, mfcc_line.shape[0]) * hopsize_t
    cax = plt.pcolormesh(x, y, np.transpose(mfcc_line[:, :, 7]))

    ax1.set_ylabel('Mel bands', fontsize=12)
    ax1.get_xaxis().set_visible(False)
    ax1.axis('tight')
    # plt.title('Calculating: '+rn+' phrase '+str(i_obs))

    ax2 = plt.subplot(212, sharex=ax1)
    x = np.arange(0, len(vad_line)) * hopsize_t
    ax2.plot(x, vad_line)
    ax2.set_ylabel('VAD', fontsize=12)
    ax2.axis('tight')
    plt.show()

if __name__ == '__main__':

    plot = False

    model_keras_cnn_0 = load_model(kerasModels_path)

    # open a pickle from python 2 in python 3, requires to add encoding
    scaler = pickle.load(open(kerasScaler_path, 'rb'), encoding='latin1')

    # the test dataset filenames
    primarySchool_val_recordings, primarySchool_test_recordings = getTestRecordingsJoint()

    for artist, fn in primarySchool_val_recordings+primarySchool_test_recordings:

        # textgrid path, to get the line onset offset
        groundtruth_textgrid_file = os.path.join(primarySchool_textgrid_path, artist, fn + '.TextGrid')

        # parse the TextGrid
        list_line = textGrid2WordList(groundtruth_textgrid_file, whichTier='line')

        wav_file = os.path.join(primarySchool_wav_path, artist, fn + '.wav')

        vad_results = VAD(wav_file)

        # calculate log mel
        log_mel = getMFCCBands2DMadmom(wav_file, fs, hopsize_t, channel=1)
        log_mel_scaled = scaler.transform(log_mel)
        log_mel_reshaped = featureReshape(log_mel_scaled, nlen=7)

        ii_line = 0
        for line in list_line: # iterate each line
            if len(line[2].strip()):

                # start and end time
                time_start = line[0]
                time_end = line[1]
                frame_start = int(round(time_start / hopsize_t))
                frame_end = int(round(time_end / hopsize_t))
                frame_end = frame_end if frame_end <= len(vad_results) else len(vad_results)

                # log_mel_reshape line
                log_mel_reshaped_line = log_mel_reshaped[frame_start: frame_end]
                log_mel_reshaped_line = np.expand_dims(log_mel_reshaped_line, axis=1)

                # emission probabilities
                obs_line = model_keras_cnn_0.predict(log_mel_reshaped_line, batch_size=128, verbose=0)
                obs_line = np.log(obs_line)

                # vad frame level
                vad_line = vad_results[frame_start: frame_end]

                if plot:
                    vad_line_for_plot = [1.0 if vl else 0 for vl in vad_line]
                    figurePlot(mfcc_line=log_mel_reshaped_line,
                               vad_line=vad_line_for_plot)

                # parse the detected phoneme onset
                onset_detection_file = os.path.join(path_jan_joint_results, artist, fn+'_'+str(ii_line)+'.pkl')
                groundtruth_detected_onset_to_load = pickle.load(open(onset_detection_file, 'rb'), encoding='bytes')
                phoneme_onset_detected = groundtruth_detected_onset_to_load[3]

                GOP_line = []
                for ii_phn in range(len(phoneme_onset_detected)):

                    # the list only contain the onset,
                    phn_start_frame = int(round(phoneme_onset_detected[ii_phn][0] / hopsize_t))

                    # the offset is the onset of the subsequence phone
                    if ii_phn < len(phoneme_onset_detected) - 1:
                        phn_end_frame = int(round(phoneme_onset_detected[ii_phn+1][0] / hopsize_t))
                    else:
                        phn_end_frame = frame_end - frame_start

                    phn_label = phoneme_onset_detected[ii_phn][1]

                    # the case of the phn length is 0
                    if phn_end_frame == phn_start_frame:
                        GOP_line.append([-np.inf, phn_label])
                        continue

                    obs_line_phn = obs_line[phn_start_frame:phn_end_frame]

                    # remove the unvoiced frames
                    if phn_label not in pho_consonants: # not remove any frames for consonants
                        vad_line_phn = vad_line[phn_start_frame:phn_end_frame]
                        if not np.all(vad_line_phn):
                            obs_line_phn = obs_line_phn[vad_line_phn, :]


                    if obs_line_phn.shape[0] == 0:
                        GOP_line.append([-np.inf, phn_label])
                        continue

                    # calculate GOP
                    GOP_phn = GOP_phn_level(phn_label= phn_label, obs_line_phn=obs_line_phn)
                    GOP_line.append([GOP_phn, phn_label])

                print(GOP_line)

                ii_line += 1

