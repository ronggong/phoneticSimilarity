import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
# import matplotlib as mpl
# mpl.style.use('classic')
from matplotlib.patches import ConnectionPatch
from src.parameters import *


def load_data(aria_name):
    plotting_data_folder = "../data/plotting_data/"
    log_mel_teacher_path = os.path.join(plotting_data_folder, "teacher_"+aria_name+".pkl")
    log_mel_student_path = os.path.join(plotting_data_folder, "student_"+aria_name+".pkl")
    simi_pronunciation_path = os.path.join(plotting_data_folder, "simi_pronunciation_"+aria_name+".pkl")
    simi_overall_quality_path = os.path.join(plotting_data_folder, "simi_overall_quality_"+aria_name+".pkl")
    extra_teacher_phns_path = os.path.join(plotting_data_folder, "extra_teacher_phns_"+aria_name+".pkl")

    log_mel_teacher, list_phns_teacher = pickle.load(open(log_mel_teacher_path, "rb"))
    log_mel_student, list_phns_student = pickle.load(open(log_mel_student_path, "rb"))
    simi_pronunciation = pickle.load(open(simi_pronunciation_path, "rb"))
    simi_overall_quality = pickle.load(open(simi_overall_quality_path, "rb"))
    extra_teacher_phns = pickle.load(open(extra_teacher_phns_path, "rb"))

    return log_mel_teacher, log_mel_student, \
           list_phns_teacher, list_phns_student, \
           simi_pronunciation, simi_overall_quality, \
           extra_teacher_phns


def score_plot(log_mel_teacher, log_mel_student,
               list_phns_teacher, list_phns_student,
               simi_pronunciation, simi_overall_quality,
               extra_teacher_phns, plot_filename):

    fontsize = 15
    text_shift = 0.002
    text_color = "r"
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

    # pro
    ax1 = plt.subplot(gs[0])
    y = np.arange(0, 80)
    x1 = np.arange(0, log_mel_teacher.shape[0]) * hopsize_t
    ax1.pcolormesh(x1, y, np.transpose(log_mel_teacher))
    for p in list_phns_teacher:
        start = p[0]-list_phns_teacher[0][0]
        plt.axvline(start, color='r', linewidth=1)
        plt.text(start+text_shift, 70, p[2], color=text_color)
    ax1.set_ylabel("Professional", fontsize=fontsize)
    ax1.axis('tight')

    # amateur
    ax2 = plt.subplot(gs[1])
    x2 = np.arange(0, log_mel_student.shape[0]) * hopsize_t
    ax2.pcolormesh(x2, y, np.transpose(log_mel_student))
    for p in list_phns_student:
        start = p[0]-list_phns_student[0][0]
        plt.axvline(start, color='r', linewidth=1)
        plt.text(start+text_shift, 70, p[2], color=text_color)
    ax2.set_ylabel("Amateur", fontsize=fontsize)
    ax2.set_xlabel("Time (s)", fontsize=fontsize)
    ax2.axis('tight')

    # inter axes arrow
    # remove extra_teacher_phns
    list_phns_teacher_no_extra = [list_phns_teacher[ii] for ii in range(len(list_phns_teacher))
                                  if ii not in extra_teacher_phns and len(list_phns_teacher[ii][2]) > 0]
    for ii in range(len(list_phns_student)):
        phn_teacher = list_phns_teacher_no_extra[ii]
        phn_student = list_phns_student[ii]
        x_phn_teacher = phn_teacher[0] - list_phns_teacher_no_extra[0][0] + (phn_teacher[1] - phn_teacher[0]) / 2.0
        x_phn_student = phn_student[0] - list_phns_student[0][0] + (phn_student[1] - phn_student[0]) / 2.0

        xy_teacher = [x_phn_teacher, 0]
        xy_student = [x_phn_student, 79]
        print(xy_teacher)
        print(xy_student)
        con = ConnectionPatch(xyA=xy_student, xyB=xy_teacher, coordsA="data", coordsB="data",
                              axesA=ax2, axesB=ax1, arrowstyle="<->", shrinkA=5, shrinkB=5)
        ax2.add_artist(con)

    # similarity
    ax3 = plt.subplot(gs[2], sharex=ax2)
    width = 0.025
    ind = np.array([phn[0]-list_phns_student[0][0]+(phn[1]-phn[0]-width)/2.0 for phn in list_phns_student])
    rects1 = ax3.bar(ind, simi_pronunciation, width, edgecolor='k', facecolor="none")
    rects2 = ax3.bar(ind+width, simi_overall_quality, width, edgecolor='k', facecolor="none", hatch="//")
    ax3.legend((rects1[0], rects2[0]),
              ('Pronunciation', 'Overall quality'),
               loc='upper center', bbox_to_anchor=(0.5, 1.1),
               fancybox=True, shadow=True, ncol=2)
    ax3.get_xaxis().set_visible(False)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("Similarity", fontsize=fontsize)
    ax2.axis('tight')

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=150)

    # plt.show()


if __name__ == "__main__":
    log_mel_teacher_yang, \
    log_mel_student_yang, \
    list_phns_teacher_yang, \
    list_phns_student_yang, \
    simi_pronunciation_yang, \
    simi_overall_quality_yang, \
    extra_teacher_phns_yang = \
        load_data("yang_yu_huan")


    num_teacher_phns = 10
    num_student_phns = 7
    num_frame_teacher = int((list_phns_teacher_yang[num_teacher_phns-1][1]-list_phns_teacher_yang[0][0])/hopsize_t)
    num_frame_student = int((list_phns_student_yang[num_student_phns-1][1]-list_phns_student_yang[0][0])/hopsize_t)
    log_mel_teacher_yang = log_mel_teacher_yang[:num_frame_teacher]
    log_mel_student_yang = log_mel_student_yang[:num_frame_student]
    score_plot(log_mel_teacher=log_mel_teacher_yang,
               log_mel_student=log_mel_student_yang,
               list_phns_teacher=list_phns_teacher_yang[:num_teacher_phns],
               list_phns_student=list_phns_student_yang[:num_student_phns],
               simi_pronunciation=simi_pronunciation_yang[:num_student_phns],
               simi_overall_quality=simi_overall_quality_yang[:num_student_phns],
               extra_teacher_phns=extra_teacher_phns_yang,
               plot_filename="../figs/scoring/yang_yu_huan.png")

    log_mel_teacher_meng, \
    log_mel_student_meng, \
    list_phns_teacher_meng, \
    list_phns_student_meng, \
    simi_pronunciation_meng, \
    simi_overall_quality_meng, \
    extra_teacher_phns_meng = \
        load_data("meng_ting_de")

    num_teacher_phns = 9
    num_student_phns = 6
    num_frame_teacher = int((list_phns_teacher_meng[num_teacher_phns-1][1]-list_phns_teacher_meng[0][0])/hopsize_t)
    num_frame_student = int((list_phns_student_meng[num_student_phns-1][1]-list_phns_student_meng[0][0])/hopsize_t)
    log_mel_teacher_meng = log_mel_teacher_meng[:num_frame_teacher]
    log_mel_student_meng = log_mel_student_meng[:num_frame_student]

    print(simi_pronunciation_meng)
    print(simi_overall_quality_meng)

    score_plot(log_mel_teacher=log_mel_teacher_meng,
               log_mel_student=log_mel_student_meng,
               list_phns_teacher=list_phns_teacher_meng[:num_teacher_phns],
               list_phns_student=list_phns_student_meng[:num_student_phns],
               simi_pronunciation=simi_pronunciation_meng[:num_student_phns],
               simi_overall_quality=simi_overall_quality_meng[:num_student_phns],
               extra_teacher_phns=extra_teacher_phns_meng,
               plot_filename="../figs/scoring/meng_ting_de.png")