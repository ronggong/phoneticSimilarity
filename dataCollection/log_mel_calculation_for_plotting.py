import os
import pickle
from src.parameters import *
from src.filePathGOPmodel import *
from src.textgridParser import syllableTextgridExtraction
from src.audio_preprocessing import getMFCCBandsMadmom


def extract_log_mel_of_line(wav_path, textgrid_path, filename, num_line):
    nestedPhonemeLists, numlines, numPhonemes \
        = syllableTextgridExtraction(textgrid_path,
                                     filename,
                                     "line",
                                     "details")

    # audio filename
    wav_full_filename = os.path.join(wav_path, filename + '.wav')

    log_mel = getMFCCBandsMadmom(audio_fn=wav_full_filename, fs=fs, hopsize_t=hopsize_t)

    line_list = nestedPhonemeLists[num_line]

    sf = int(round(line_list[0][0] * fs / float(hopsize)))  # starting frame
    ef = int(round(line_list[0][1] * fs / float(hopsize)))  # ending frame

    return (log_mel[sf:ef], line_list[1])


if __name__ == "__main__":
    teacher_filename_yang_yu_huan = "20171214SongRuoXuan/daeh-Yang_yu_huan-Tai_zhen_wai_zhuan-nanluo/teacher"
    student_filename_yang_yu_huan = "20171214SongRuoXuan/daeh-Yang_yu_huan-Tai_zhen_wai_zhuan-nanluo/student_01"

    teacher_filename_meng_ting_de = "20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky/teacher"
    student_filename_meng_ting_de = "20171211SongRuoXuan/daxp-Meng_ting_de-Mu_gui_ying_gua_shuai-dxjky/student03"

    plotting_data_folder = "../data/plotting_data/"

    teacher_yang_yu_huan = extract_log_mel_of_line(wav_path=primarySchool_wav_path,
                                                   textgrid_path=primarySchool_textgrid_path,
                                                   filename=teacher_filename_yang_yu_huan,
                                                   num_line=0)
    pickle.dump(teacher_yang_yu_huan, open(os.path.join(plotting_data_folder, "teacher_yang_yu_huan.pkl"), "wb"))
    student_yang_yu_huan = extract_log_mel_of_line(wav_path=primarySchool_wav_path,
                                                   textgrid_path=primarySchool_textgrid_path,
                                                   filename=student_filename_yang_yu_huan,
                                                   num_line=0)
    pickle.dump(student_yang_yu_huan, open(os.path.join(plotting_data_folder, "student_yang_yu_huan.pkl"), "wb"))

    teacher_meng_ting_de = extract_log_mel_of_line(wav_path=primarySchool_wav_path,
                                                   textgrid_path=primarySchool_textgrid_path,
                                                   filename=teacher_filename_meng_ting_de,
                                                   num_line=0)
    pickle.dump(teacher_meng_ting_de, open(os.path.join(plotting_data_folder, "teacher_meng_ting_de.pkl"), "wb"))
    student_meng_ting_de = extract_log_mel_of_line(wav_path=primarySchool_wav_path,
                                                   textgrid_path=primarySchool_textgrid_path,
                                                   filename=student_filename_meng_ting_de,
                                                   num_line=0)
    pickle.dump(student_meng_ting_de, open(os.path.join(plotting_data_folder, "student_meng_ting_de.pkl"), "wb"))
