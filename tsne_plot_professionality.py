from sklearn.manifold import TSNE
import os
import pickle
import itertools
import numpy as np
from parameters import config_select

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1,
                rc={"lines.linewidth": 2.5})

# from keras.models import Model


def plot_tsne_profess(embeddings, labels, le):
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=3000)
    for ii_class in range(27):
        len_stu = len(embeddings[labels == 2*ii_class, :])
        label_class_name0 = le.inverse_transform(2 * ii_class)
        label_class_name1 = le.inverse_transform(2 * ii_class+1)

        # plot t_sne for teacher and student
        try:
            tsne_results = tsne.fit_transform(np.vstack((embeddings[labels == 2*ii_class, :],
                                                         embeddings[labels == 2*ii_class+1, :])))
            plt.figure()

            plt.scatter(tsne_results[:len_stu, 0],
                        tsne_results[:len_stu, 1],
                        label=label_class_name0)

            plt.scatter(tsne_results[len_stu:, 0],
                        tsne_results[len_stu:, 1],
                        label=label_class_name1,
                        marker='v')
            plt.legend()
            plt.savefig(os.path.join('./figs/professionality/MTL/', label_class_name0.split('_')[0]+'.png'),
                        bbox_inches='tight')
            # plt.show()
        except:
            pass


def plot_tsne_pronun(embeddings, labels, le):

    markers = ['.', ',', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd']
    marker_obj = itertools.cycle(markers)

    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=3000)
    tsne_results = tsne.fit_transform(embeddings)
    # new_cmap = rand_cmap(29, type='bright', first_color_black=True, last_color_black=False, verbose=False)
    palette = np.array(sns.hls_palette(54, l=.4, s=.8))

    plt.figure()

    for ii_class in range(0, 54, 4):
        label_class_name0 = le.inverse_transform(ii_class)
        plt.scatter(tsne_results[labels == ii_class, 0],
                    tsne_results[labels == ii_class, 1],
                    # color=new_cmap(ii_class),
                    c=palette[ii_class],
                    label=label_class_name0.split('_')[0],
                    marker=next(marker_obj),
                    s=50)

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    for ii_class in range(0, 54, 4):
        x = tsne_results[labels == ii_class, 0]
        y = tsne_results[labels == ii_class, 1]
        content = le.inverse_transform(ii_class).split('_')[0]
        for ii_text in range(0, len(x), 5):
            plt.annotate(content, (x[ii_text], y[ii_text]), color=palette[ii_class])
            # plt.text(x[ii_text], y[ii_text], content, color=new_cmap(ii_class))

    # # We add the labels for each digit.
    # txts = []
    # for ii_class in range(29):
    #     # Position of each label.
    #     xtext, ytext = np.median(tsne_results[labels==ii_class, :], axis=0)
    #     content = labels_original[labels==ii_class][0]
    #     txt = plt.text(xtext, ytext, content, fontsize=24)
    #     # txt.set_path_effects([
    #     #     PathEffects.Stroke(linewidth=5, foreground="w"),
    #     #     PathEffects.Normal()])
    #     txts.append(txt)


    plt.show()


if __name__ == '__main__':

    MTL = True
    config = [2, 0]
    embedding_dim = 2
    path_eval = '/home/gong/Documents/pycharmProjects/phoneticSimilarity/eval/phone_embedding_classifier'

    prefix = '_MTL' if MTL else '_2_class_teacher_student'
    model_name = config_select(config) + prefix if embedding_dim == 2 else config_select(config)

    le = pickle.load(open(os.path.join(path_eval, model_name + '_le.pkl'), 'rb'))
    embedding_profess = np.load(os.path.join(path_eval, model_name + '_embedding_professionality0.npy'))
    embedding_pronun = np.load(os.path.join(path_eval, model_name + '_embedding_pronunciation0.npy'))
    labels = np.load(os.path.join(path_eval, model_name + '_embeddings_labels.npy'))

    plot_tsne_pronun(embedding_pronun, labels, le)
    # plot_tsne_profess(embedding_profess, labels, le)