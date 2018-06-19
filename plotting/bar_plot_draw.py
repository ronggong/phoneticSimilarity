"""
plot the bar charts
"""

import matplotlib.pyplot as plt
import numpy as np


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.3f' % height,
                ha='center', va='bottom')

N = 2
ind = np.arange(N)
width = 0.25

mean_baseline_cal = [0.6452855784255096, 0.6318295940979021]
std_baseline_cal = [0.0077447490877095935, 0.011464005989563009]

fig, ax = plt.subplots()
rects1 = ax.bar(ind, mean_baseline_cal, width, color='#990000', yerr=std_baseline_cal)

mean_baseline_sia = [0.37264827364469, 0.5105292151303018]
std_baseline_sia = [0.021014488169857657, 0.002939691091241116]

rects2 = ax.bar(ind+width, mean_baseline_sia, width, color='#ffcc00', yerr=std_baseline_sia, hatch="+")

mean_baseline_sia_random = [0.263075708680922, 0.5191045405028277]
std_baseline_sia_random = [0.01360165046893119, 0.00925388680804896]

rects3 = ax.bar(ind+2*width, mean_baseline_sia_random, width, color='#660099', yerr=std_baseline_sia_random, hatch="//")

ax.set_ylabel('Average precision', fontsize=15)
# ax.set_title('Baseline classification and Siamese networks evaluation results')
ax.set_xticks(ind + width)
ax.set_xticklabels(('Pronunciation', 'Overall quality'), fontsize=15)
ax.set_ylim(0, 0.82)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.legend((rects1[0], rects2[0], rects3[0]), ('Classification net\nBaseline', 'Siamese net', 'Siamese net\nrandom weights'),
          loc='upper center', bbox_to_anchor=(0.5, 1),
          fancybox=False, shadow=False, ncol=3)
plt.tight_layout()

plt.show()


N = 2
ind = np.arange(N)
width = 0.15

fig, ax = plt.subplots()

mean_baseline = [0.6452855784255096, 0.6318295940979021]
std_baseline = [0.0077447490877095935, 0.011464005989563009]

rects1 = ax.bar(ind, mean_baseline, width, edgecolor='k', facecolor="none", yerr=std_baseline, hatch="o")

mean_att = [0.6568787911883697, 0.6115934685913207]
std_att = [0.004155328527582738, 0.01608948287744057]

rects2 = ax.bar(ind+width, mean_att, width, edgecolor='k', facecolor="none", yerr=std_att, hatch="+")

mean_32 = [0.5991873660106768, 0.6749754070227755]
std_32 = [0.0049621182346896685, 0.024750238114994057]

rects3 = ax.bar(ind+2*width, mean_32, width, edgecolor='k', facecolor="none", yerr=std_32, hatch="//")

mean_cnn = [0.7239767493545097, 0.6548589730636626]
std_cnn = [0.005315364966810669, 0.011330924368280058]

rects4 = ax.bar(ind+3*width, mean_cnn, width, edgecolor='k', facecolor="none", yerr=std_cnn, hatch="*")

mean_dropout = [0.6797722536054238, 0.6299049968434]
std_dropout = [0.004984889157997945, 0.003506507902422659]

rects5 = ax.bar(ind+4*width, mean_dropout, width, edgecolor='k', facecolor='none', yerr=std_dropout, hatch="/")

mean_best_comb = [0.7534669090268483, 0.6727435374437956]
std_best_comb = [0.005871469344325567, 0.026698363540360672]

rects6 = ax.bar(ind+5*width, mean_best_comb, width, edgecolor='k', facecolor="none", yerr=std_best_comb, hatch="x")

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)

ax.set_ylabel('Average precision', fontsize=15)
ax.set_ylim(0, 0.9)
# ax.set_title('Improving classification network')
ax.set_xticks(ind + 5*width / 2.0)
ax.set_xticklabels(('Pronunciation', 'Overall quality'), fontsize=15)

ax.legend((rects1[0], rects2[0], rects3[0], rects4[0], rects5[0], rects6[0]),
          ('Baseline', 'Attention', '32 embedding', 'CNN', 'Dropout', 'best\ncombination'),
          loc='upper center', bbox_to_anchor=(0.5, 1),
          fancybox=False, shadow=False, ncol=6)

plt.tight_layout()

plt.show()


