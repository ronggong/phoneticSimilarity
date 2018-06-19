import os
import pickle
import numpy as np
import h5py
from sklearn.feature_selection import f_classif
from ANOVA_exp.constants import index_151_in_152
from ANOVA_exp.constants import phonemes
import matplotlib.pyplot as plt


def feature_list_2_mat(root_phn_wav_path, sub_folders):
    desc_name_final = None  # descriptor names
    desc_name_152 = None
    desc_name_151 = None
    y = {}  # ground truth
    dict_feature = {}
    for p in phonemes:
        dict_feature[p] = []
        y[p] = []
    for ii, folder in enumerate(sub_folders):
        path_phn_feature = os.path.join(root_phn_wav_path, folder)
        filenames_feature = [f for f in os.listdir(path_phn_feature) if
                             os.path.isfile(os.path.join(path_phn_feature, f))]
        for jj, fn in enumerate(filenames_feature):
            key = fn.split('_')[0]
            if key == "?" or key == "sil":
                continue
            with open(os.path.join(path_phn_feature, fn), 'r') as f:
                feature, desc_name = pickle.load(f)
                # if len(desc_name) == 152:
                #     desc_name_152 = desc_name
                # elif len(desc_name) == 151:
                #     desc_name_151 = desc_name
                # if desc_name_152 and desc_name_151:
                #     index_151_in_152 = [desc_name_152.index(d) for d in desc_name_151]
                #     print(index_151_in_152)
                #     break
                if len(feature) == 152:
                    # some feature dim could be 676, reduce it to 656
                    feature = feature[index_151_in_152]
                else:
                    desc_name_final = desc_name
                    print(desc_name_final)
                    break
                dict_feature[key].append(feature)
                y[key].append(ii)
            print("loading {} {} feature out of {}, feature length {}, key {}".format(fn, jj, len(filenames_feature),
                                                                              len(feature), key))

    for p in phonemes:
        dict_feature[p] = np.vstack(dict_feature[p])
        with open("./data/"+p+".pkl", "wb") as f:
            pickle.dump([dict_feature[p], y[p], desc_name_final], f)
    # with h5py.File("./data/feature.hdf5", "w") as f:
    #     f.create_dataset("feature", [dict_feature, y, desc_name_656])


def box_plot(data, feature_name):
    # notched plot
    fig = plt.figure()
    plt.boxplot(data, 1, "r+")
    plt.ylabel("Value distribution", fontsize=15)
    plt.xticks([1, 2, 3], ["Professional", "Amateur\ntrain val", "Amateur\ntest"], fontsize=15)
    # plt.title(feature_name)
    plt.tight_layout()
    return fig


if __name__ == '__main__':

    root_phn_wav_path = "/Volumes/rong_segate/phoneme_audio_dlfm"
    sub_folders = ['teacher_feature', 'student_feature', 'extra_test_feature']
    # feature_list_2_mat(root_phn_wav_path=root_phn_wav_path,
    #                    sub_folders=sub_folders)
    # f = h5py.File("./data/feature.hdf5", "r")
    # feature, y, desc_name = f["feature"]
    # print(feature)

    phonemes_spec = ["y", "nvc", "in", "O"]
    for p in phonemes_spec:
        with open("./data/"+p+".pkl", "rb") as f:
            feature, y, desc_name = pickle.load(f)
        y = np.array(y)
        F_val, p_val = f_classif(feature, y)
        sort_index = np.argsort(F_val)[::-1]
        sort_desc_name = [desc_name[ii] for ii in sort_index]

        counter = 0
        for ii in range(len(sort_index)):
            if "start" in sort_desc_name[ii] or \
                    "silence" in sort_desc_name[ii] or \
                    "stop" in sort_desc_name[ii] or \
                    "ebu128" in sort_desc_name[ii] or \
                    "rhythm" in sort_desc_name[ii] or \
                    "chords" in sort_desc_name[ii] or \
                    "effective_duration" in sort_desc_name[ii]:
                continue
            ind = sort_index[ii]
            feature_teacher = feature[np.where(y == 0)[0], ind]
            feature_student = feature[np.where(y == 1)[0], ind]
            feature_extra_test = feature[np.where(y == 2)[0], ind]
            fig = box_plot([feature_teacher, feature_student, feature_extra_test], sort_desc_name[ii])
            filename = "./plot/"+p+"_"+str(ii)+"_"+sort_desc_name[ii]+".png"
            fig.savefig(filename, dpi=200)
            counter += 1
            if counter >= 4:
                break
