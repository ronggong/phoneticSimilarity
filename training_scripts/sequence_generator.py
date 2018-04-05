import numpy as np
from keras.utils.data_utils import Sequence


class tripletNdiffYieldIndexSequence(Sequence):

    def __init__(self, list_feature, labels, batch_size=1, N_diff=5):
        self.list_feature, self.labels = list_feature, labels
        self.batch_size = batch_size
        self.N_diff = N_diff

    def __len__(self):
        return int(np.ceil(len(self.list_feature) / float(self.batch_size)))

    def __getitem__(self, index):

        out = []
        for ii in range(index*self.batch_size, (index+1)*self.batch_size):

            # in case of ii is too big
            if ii >= len(self.list_feature):
                ii -= len(self.list_feature)

            anchor = self.list_feature[ii]

            # random select a same sample
            idx_same = np.where(self.labels == self.labels[ii])[0]
            idx_same = idx_same[idx_same != ii]
            same = self.list_feature[idx_same[np.random.choice(len(idx_same), size=1)][0]]

            # random select a diff sample
            idx_diff = np.where(self.labels != self.labels[ii])[0]

            out_sample = []
            for jj_ndiff in range(self.N_diff):
                diff = self.list_feature[idx_diff[np.random.choice(len(idx_diff), size=1)][0]]
                out_sample.append([anchor, same, diff])

            out.append(out_sample)

        return out

    def on_epoch_end(self):
        # shuffle the data on epoch end
        p = np.random.permutation(len(self.list_feature))
        self.list_feature = [self.list_feature[ii_p] for ii_p in p]
        self.labels = self.labels[p]  # labels is a numpy array