import os

def sterero2Mono(audio_data):
    if len(audio_data.shape) == 2:
        if audio_data.shape[1] == 2:
            audio_data = (audio_data[:, 0] + audio_data[:, 1]) / 2.0
    return audio_data


def append_or_write(eval_result_file_name):
    if os.path.exists(eval_result_file_name):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not
    return append_write