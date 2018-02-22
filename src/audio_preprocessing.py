from madmom.processors import SequentialProcessor
from src.Fprev_sub import Fprev_sub
from src.Fdeltas import Fdeltas
import soundfile as sf
import resampy
import os
import wave
import contextlib
import numpy as np
import webrtcvad
import librosa


EPSILON = np.spacing(1)

def _nbf_2D(mfcc, nlen):
    mfcc = np.array(mfcc).transpose()
    mfcc_out = np.array(mfcc, copy=True)
    for ii in range(1, nlen + 1):
        mfcc_right_shift = Fprev_sub(mfcc, w=ii)
        mfcc_left_shift = Fprev_sub(mfcc, w=-ii)
        # print(mfcc_left_shift.shape, mfcc_right_shift.shape)
        mfcc_out = np.vstack((mfcc_right_shift, mfcc_out, mfcc_left_shift))
    # print(mfcc_out.shape)
    feature = mfcc_out.transpose()
    return feature


class MadmomMelbankProcessor(SequentialProcessor):


    def __init__(self, fs, hopsize_t):
        from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
        from madmom.audio.stft import ShortTimeFourierTransformProcessor
        from madmom.audio.filters import MelFilterbank
        from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                              LogarithmicSpectrogramProcessor)
        # from madmom.features.onsets import _cnn_onset_processor_pad

        # define pre-processing chain
        sig = SignalProcessor(num_channels=1, sample_rate=fs)
        # process the multi-resolution spec in parallel
        # multi = ParallelProcessor([])
        # for frame_size in [2048, 1024, 4096]:
        frames = FramedSignalProcessor(frame_size=2048, hopsize=int(fs*hopsize_t))
        stft = ShortTimeFourierTransformProcessor()  # caching FFT window
        filt = FilteredSpectrogramProcessor(
            filterbank=MelFilterbank, num_bands=80, fmin=27.5, fmax=16000,
            norm_filters=True, unique_filters=False)
        spec = LogarithmicSpectrogramProcessor(log=np.log, add=EPSILON)

        # process each frame size with spec and diff sequentially
        # multi.append())
        single = SequentialProcessor([frames, stft, filt, spec])

        # stack the features (in depth) and pad at beginning and end
        # stack = np.dstack
        # pad = _cnn_onset_processor_pad

        # pre-processes everything sequentially
        pre_processor = SequentialProcessor([sig, single])

        # instantiate a SequentialProcessor
        super(MadmomMelbankProcessor, self).__init__([pre_processor])


def getMFCCBands2DMadmom(audio_fn, fs, hopsize_t, channel):
    """
    extract the log mel spectrogram
    :param audio_fn:
    :param fs:
    :param hopsize_t:
    :param channel:
    :return:
    """

    mfcc = getMFCCBandsMadmom(audio_fn=audio_fn, fs=fs, hopsize_t=hopsize_t)
    if channel == 1:
        mfcc = _nbf_2D(mfcc, 7)
    else:
        mfcc_conc = []
        for ii in range(3):
            mfcc_conc.append(_nbf_2D(mfcc[:,:,ii], 7))
        mfcc = np.stack(mfcc_conc, axis=2)
    return mfcc


def getMFCCBandsMadmom(audio_fn, fs, hopsize_t):
    madmomMelbankProc = MadmomMelbankProcessor(fs, hopsize_t)
    mfcc = madmomMelbankProc(audio_fn)
    return mfcc


def featureReshape(feature, nlen=10):
    """
    reshape mfccBands feature into n_sample * n_row * n_col
    :param feature:
    :return:
    """

    n_sample = feature.shape[0]
    n_row = 80
    n_col = nlen*2+1

    feature_reshaped = np.zeros((n_sample,n_row,n_col),dtype='float32')
    # print("reshaping feature...")
    for ii in range(n_sample):
        # print ii
        feature_frame = np.zeros((n_row,n_col),dtype='float32')
        for jj in range(n_col):
            feature_frame[:,jj] = feature[ii][n_row*jj:n_row*(jj+1)]
        feature_reshaped[ii,:,:] = feature_frame
    return feature_reshaped


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, hopsize_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    framesize = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    hopsize = int(sample_rate * (hopsize_ms / 1000.0) *2)
    offset = 0
    timestamp = 0.0
    offfset_timestamp = (float(hopsize) / sample_rate) / 2.0
    duration = (float(framesize) / sample_rate) / 2.0
    while offset + framesize < len(audio):
        yield Frame(audio[offset:offset + framesize], timestamp, duration)
        timestamp += offfset_timestamp
        offset += hopsize


def VAD(wav_file):
    resample_fs = 32000
    current_path = os.path.dirname(os.path.realpath(__file__))
    path_temp_wav = os.path.join(current_path, '..', 'temp', 'temp.wav')
    wav_data, wav_fs = sf.read(wav_file)
    vad_results = np.array([], dtype=np.bool)

    # convert the audio to the 1 channel
    if len(wav_data.shape) == 2:
        if wav_data.shape[1] == 2:
            wav_data = (wav_data[:, 0] + wav_data[:, 1]) / 2.0

    # resample the audio samples
    wav_data_32000 = resampy.resample(wav_data, wav_fs, resample_fs)

    # write the audio
    sf.write(path_temp_wav, wav_data_32000, resample_fs)

    # read the wav in bytes, the length will be 2 times of the normal wav data
    wav_data, wav_fs = read_wave(path_temp_wav)

    # gnerate frames
    frames = frame_generator(frame_duration_ms=30, hopsize_ms=10, audio=wav_data, sample_rate=wav_fs)

    vad = webrtcvad.Vad()

    # mode 0-3, 3 is the most aggressive one
    vad.set_mode(0)
    for frame in frames:
        is_speech = vad.is_speech(buf=frame.bytes, sample_rate=wav_fs)
        vad_results = np.append(vad_results, is_speech)

    # print(wav_data_32000.shape)

    return vad_results


def mfccDeltaDelta(audio_data, fs, framesize, hopsize):
    """
    mfcc and derivative, second derivative
    :param audio_data:
    :param fs:
    :param framesize:
    :param hopsize:
    :return:
    """
    mfccs = librosa.feature.mfcc(y=audio_data, sr=fs, n_mfcc=13, n_fft=framesize, hop_length=hopsize)
    mfccs_delta = Fdeltas(x = mfccs, w=5)
    mfccs_delta_delta = Fdeltas(x = mfccs_delta, w=5)
    out = np.vstack((mfccs, mfccs_delta, mfccs_delta_delta))
    return out


def segmentMfccLine(line, hopsize_t, mfccs):
    """
    segment line level mfccs
    :param line: [start_time, end_time, lyrics]
    :return:
    """
    # start and end time
    time_start = line[0]
    time_end = line[1]
    frame_start = int(round(time_start / hopsize_t))
    frame_end = int(round(time_end / hopsize_t))

    # log_mel_reshape line
    mfccs_line = mfccs[:, frame_start: frame_end]
    return mfccs_line