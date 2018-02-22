import numpy as np
import h5py
import itertools

from keras.utils.np_utils import to_categorical

def shuffleFilenamesLabelsInUnison(filenames, labels):
    assert len(filenames) == len(labels)
    p=np.random.permutation(len(filenames))
    return filenames[p], labels[p]

def generator(path_feature_data,
              indices,
              number_of_batches,
              file_size,
              input_shape,
              labels=None,
              shuffle=True,
              multi_inputs=False,
              channel=1):

    # print(len(filenames))
    # print(path_feature_data)
    f = h5py.File(path_feature_data, 'r')
    indices_copy = np.array(indices[:], np.int64)

    if labels is not None:
        labels_copy = np.copy(labels)
        labels_copy = to_categorical(labels_copy)
    else:
        labels_copy = np.zeros((len(indices_copy), ))

    counter = 0

    while True:
        idx_start = file_size * counter
        idx_end = file_size * (counter + 1)

        batch_indices = indices_copy[idx_start:idx_end]
        index_sort = np.argsort(batch_indices)

        y_batch_tensor = labels_copy[idx_start:idx_end][index_sort]

        if channel == 1:
            X_batch_tensor = f['feature_all'][batch_indices[index_sort],:,:]
        else:
            X_batch_tensor = f['feature_all'][batch_indices[index_sort], :, :, :]
        if channel == 1:
            X_batch_tensor = np.expand_dims(X_batch_tensor, axis=1)

        counter += 1

        if multi_inputs:
            yield [X_batch_tensor,X_batch_tensor,X_batch_tensor,X_batch_tensor,X_batch_tensor,X_batch_tensor], y_batch_tensor
        else:
            yield X_batch_tensor, y_batch_tensor

        if counter >= number_of_batches:
            counter = 0
            if shuffle:
                indices_copy, labels_copy = shuffleFilenamesLabelsInUnison(indices_copy, labels_copy)


def sort_feature_by_seq_length(list_feature, labels, batch_size):
    # sort the list_feature and the labels by the length
    list_len = [len(l) for l in list_feature]
    idx_sort = np.argsort(list_len)
    list_feature_sorted = [list_feature[ii] for ii in idx_sort]
    labels_sorted = labels[idx_sort]

    iter_times = int(np.ceil(len(list_feature_sorted) / float(batch_size)))

    return list_feature_sorted, labels_sorted, iter_times


def batch_grouping(list_feature_sorted, labels_sorted, iter_times, batch_size):
    """
    group the features and labels into batches,
    each feature batch has more or less the sequence with the similar length
    :param list_feature:
    :param labels:
    :param batch_size:
    :return:
    """

    # aggregate the iter_times-1 batch
    list_X_batch = []
    list_y_batch = []
    for ii in range(iter_times-1):
        max_len_in_batch = list_feature_sorted[(ii+1)*batch_size-1].shape[0]
        feature_dim_in_batch = list_feature_sorted[(ii+1)*batch_size-1].shape[1]

        X_batch = np.zeros((batch_size, max_len_in_batch, feature_dim_in_batch), dtype='float32')
        y_batch = labels_sorted[ii*batch_size: (ii+1)*batch_size, :]

        # print(ii*batch_size, (ii+1)*batch_size)
        for jj in range(ii*batch_size, (ii+1)*batch_size):
            X_batch[jj-ii*batch_size, :len(list_feature_sorted[jj]), :] = list_feature_sorted[jj]

        list_X_batch.append(X_batch)
        list_y_batch.append(y_batch)

    # aggregate the last batch
    max_len_in_batch = list_feature_sorted[-1].shape[0]
    feature_dim_in_batch = list_feature_sorted[-1].shape[1]

    X_batch = np.zeros((batch_size, max_len_in_batch, feature_dim_in_batch), dtype='float32')
    y_batch = labels_sorted[-batch_size:, :]

    for jj in range(len(list_feature_sorted)-batch_size, len(list_feature_sorted)):
        X_batch[jj-(len(list_feature_sorted)-batch_size), :len(list_feature_sorted[jj]), :] = list_feature_sorted[jj]

    list_X_batch.append(X_batch)
    list_y_batch.append(y_batch)

    return list_X_batch, list_y_batch


def shuffleListBatch(list_X_batch, list_y_batch):
    assert len(list_X_batch) == len(list_y_batch)
    p = np.random.permutation(len(list_X_batch))
    list_X_batch = [list_X_batch[ii] for ii in p]
    list_y_batch = [list_y_batch[ii] for ii in p]

    for ii_batch in range(len(list_X_batch)):
        p = np.random.permutation(list_X_batch[ii_batch].shape[0])
        list_X_batch[ii_batch] = list_X_batch[ii_batch][p, :, :]
        list_y_batch[ii_batch] = list_y_batch[ii_batch][p, :]


    return list_X_batch, list_y_batch


def generator_batch_group(list_X_batch,
                          list_y_batch,
                          iter_times,
                          shuffle=True):

    ii = 0
    while True:
        yield list_X_batch[ii], list_y_batch[ii]

        ii += 1
        if ii >= len(list_X_batch):
            ii = 0
            if shuffle:
                list_X_batch, list_y_batch = shuffleListBatch(list_X_batch=list_X_batch,
                                                              list_y_batch=list_y_batch)


def generator_triplet(list_feature, labels, batch_size=1, shuffle=True, reverse_anchor=False):
    """triplet pairs generator, reverse anchor and same samples"""
    ii = 0
    while True:
        anchor = list_feature[ii]

        # random select a same sample
        idx_same = np.where(labels == labels[ii])[0]
        idx_same = idx_same[idx_same != ii]
        same = list_feature[idx_same[np.random.choice(len(idx_same), size=1)][0]]

        # random select a diff sample
        idx_diff = np.where(labels != labels[ii])[0]
        diff = list_feature[idx_diff[np.random.choice(len(idx_diff), size=1)][0]]

        if batch_size == 1:
            idx_reverse = [0, 1] if reverse_anchor else [0]

            for jj_reverse in idx_reverse:
                if jj_reverse == 0:
                    yield ({'anchor_input': anchor,
                           'same_input': same,
                           'diff_input': diff},
                           None)
                elif jj_reverse == 1:
                    yield ({'anchor_input': same,
                            'same_input': anchor,
                            'diff_input': diff},
                           None)
                else:
                    pass
        else:
            raise ValueError

        ii += 1

        if ii >= len(list_feature):
            ii = 0
            if shuffle:
                p = np.random.permutation(len(list_feature))
                list_feature = [list_feature[ii_p] for ii_p in p]
                labels = labels[p] # labels is a numpy array


def generator_triplet_Ndiff(list_feature, labels, batch_size=1, shuffle=True, reverse_anchor=False, N_diff=5):
    """triplet pairs generator, reverse anchor and same samples, N different samples"""
    ii = 0
    while True:
        anchor = list_feature[ii]

        # random select a same sample
        idx_same = np.where(labels == labels[ii])[0]
        idx_same = idx_same[idx_same != ii]
        same = list_feature[idx_same[np.random.choice(len(idx_same), size=1)][0]]

        # random select a diff sample
        idx_diff = np.where(labels != labels[ii])[0]

        if batch_size == 1:
            idx_reverse = [0, 1] if reverse_anchor else [0]
            for jj_reverse in idx_reverse:
                for jj_ndiff in range(N_diff):
                    diff = list_feature[idx_diff[np.random.choice(len(idx_diff), size=1)][0]]
                    if jj_reverse == 0:
                        yield [anchor, same, diff]
                    elif jj_reverse == 1:
                        yield [same, anchor, diff]
                    else:
                        pass
        else:
            raise ValueError

        ii += 1

        if ii >= len(list_feature):
            ii = 0
            if shuffle:
                p = np.random.permutation(len(list_feature))
                list_feature = [list_feature[ii_p] for ii_p in p]
                labels = labels[p] # labels is a numpy array



def calculate_num_idx_same_pairs(labels, num_class):
    """number and index of same pairs for each phone class"""
    idx_labels = np.arange(len(labels))

    num_same_pairs = []
    idx_same_pairs = []
    for ii in range(num_class):
        idx_same_pairs_ii = np.asarray(list(itertools.permutations(idx_labels[labels == ii], r=2)))
        num_same_pairs.append(len(idx_same_pairs_ii))
        idx_same_pairs.append(idx_same_pairs_ii)

    return num_same_pairs, idx_same_pairs


def calculate_reduced_same_pairs(class_size, idx_same_pairs):
    """reduced same pairs where each phone class has the minimum and equal pair number"""

    idx_same_pairs_reduced = []
    for idx_same_pairs_ii in idx_same_pairs:
        idx_same_pairs_ii = idx_same_pairs_ii[np.random.choice(len(idx_same_pairs_ii), size=class_size)]
        idx_same_pairs_reduced += list(idx_same_pairs_ii)

    return idx_same_pairs_reduced


def generator_triplet_pairs(list_feature, labels, class_size, idx_same_pairs, batch_size=1, shuffle=True):

    idx_same_pairs_reduced = calculate_reduced_same_pairs(class_size, idx_same_pairs)
    # reduced index
    # idx_reduced = np.asarray(list(set([item for sublist in idx_same_pairs_reduced for item in sublist])))

    ii = 0
    while True:
        if ii==0 and shuffle:
            p = np.random.permutation(len(idx_same_pairs_reduced))
            idx_same_pairs_reduced = [idx_same_pairs_reduced[ii_p] for ii_p in p]

        idx_same_pair_ii = idx_same_pairs_reduced[ii]
        anchor = list_feature[idx_same_pair_ii[0]]

        # random select a same sample
        same = list_feature[idx_same_pair_ii[1]]

        # random select a diff sample
        idx_diff = np.where(labels != labels[idx_same_pair_ii[0]])[0]
        # select only from the reduced index
        # idx_diff_reduced = np.intersect1d(idx_reduced, idx_diff)
        diff = list_feature[idx_diff[np.random.choice(len(idx_diff), size=1)][0]]

        # print(labels[idx_same_pair_ii[0]], labels[idx_same_pair_ii[1]],
        #       labels[idx_diff_reduced[np.random.choice(len(idx_diff_reduced), size=1)][0]])

        if batch_size == 1:
            yield ({'anchor_input': anchor,
                   'same_input': same,
                   'diff_input': diff},
                   None)
        else:
            raise ValueError

        ii += 1

        if ii >= len(idx_same_pairs_reduced):
            ii = 0

            idx_same_pairs_reduced = calculate_reduced_same_pairs(class_size, idx_same_pairs)
            # idx_reduced = np.asarray(list(set([item for sublist in idx_same_pairs_reduced for item in sublist])))
