from keras.models import Input
from keras.models import Model
from keras.models import save_model
from keras.layers import Dropout
from keras.layers import LSTM
# from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint

import numpy as np

from feature_generator import generator_triplet
from feature_generator import generator_triplet_Ndiff
from feature_generator import calculate_num_idx_same_pairs
from feature_generator import generator_triplet_pairs

from data_preparation import writeValLossCsv
from losses import triplet_loss
# from tensorflow.python.client import device_lib



def embedding_model_base(input_shape):
    # embedding base model
    base_input = Input(batch_shape=input_shape)

    # if device == 'CPU':
    x = Bidirectional(LSTM(units=32, return_sequences=False))(base_input)

    # else:
    #     x = Bidirectional(CuDNNLSTM(units=32, return_sequences=False))(base_input)

    x = Dense(units=64, activation='relu')(x)

    x = Dropout(rate=0.5)(x)

    x = Dense(29, activation='linear', name='embedding_layer')(x)

    embedding_model = Model(inputs=base_input, outputs=x, name='embedding')

    # inputs
    anchor_input = Input(batch_shape=input_shape, name='anchor_input')
    same_input = Input(batch_shape=input_shape, name='same_input')
    diff_input = Input(batch_shape=input_shape, name='diff_input')

    anchor_embedding = embedding_model(anchor_input)
    same_embedding = embedding_model(same_input)
    diff_embedding = embedding_model(diff_input)

    inputs = [anchor_input, same_input, diff_input]
    outputs = [anchor_embedding, same_embedding, diff_embedding]

    triplet_model = Model(inputs, outputs)

    return embedding_model, triplet_model, outputs


def embedding_siamese_1_lstm_1_dense(input_shape):
    """use keras compile"""
    # device = device_lib.list_local_devices()[0].device_type
    embedding_model, triplet_model, outputs = embedding_model_base(input_shape)

    triplet_model.add_loss(K.mean(triplet_loss(outputs, margin=0.15)))
    triplet_model.compile(loss=None, optimizer='adam')

    return embedding_model, triplet_model


def embedding_model_base_2_lstm_1_dense_base(input_shape, output_shape):
    # embedding base model
    base_input = Input(batch_shape=input_shape)

    # if device == 'CPU':
    x = Bidirectional(LSTM(units=32, return_sequences=True))(base_input)
    x = Bidirectional(LSTM(units=32, return_sequences=False))(x)

    # else:
    #     x = Bidirectional(CuDNNLSTM(units=32, return_sequences=False))(base_input)

    x = Dense(units=64, activation='relu')(x)

    x = Dropout(rate=0.5)(x)

    x = Dense(output_shape, activation='linear', name='embedding_layer')(x)

    embedding_model = Model(inputs=base_input, outputs=x, name='embedding')

    # inputs
    anchor_input = Input(batch_shape=input_shape, name='anchor_input')
    same_input = Input(batch_shape=input_shape, name='same_input')
    diff_input = Input(batch_shape=input_shape, name='diff_input')

    anchor_embedding = embedding_model(anchor_input)
    same_embedding = embedding_model(same_input)
    diff_embedding = embedding_model(diff_input)

    inputs = [anchor_input, same_input, diff_input]
    outputs = [anchor_embedding, same_embedding, diff_embedding]

    triplet_model = Model(inputs, outputs)

    return embedding_model, triplet_model, outputs


def embedding_siamese_2_lstm_1_dense_model_compile(input_shape, output_shape, margin):
    """use keras compile"""
    # device = device_lib.list_local_devices()[0].device_type
    embedding_model, triplet_model, outputs = embedding_model_base_2_lstm_1_dense_base(input_shape, output_shape)

    triplet_model.add_loss(K.mean(triplet_loss(outputs, margin=margin)))
    triplet_model.compile(loss=None, optimizer='adam')

    return embedding_model, triplet_model


def calculate_loss(triplet_model,
                   generator,
                   iter_time,
                   batch_size,
                   N_diff,
                   margin):
    """calculate the max loss during Ndiff iterations"""
    max_loss = -np.inf
    ii_Ndiff = 0
    list_loss = []
    ii_counter = 0
    for input_batch in generator:
        outputs_batch = triplet_model.predict_on_batch(input_batch)
        loss_batch = K.eval(K.mean(triplet_loss(outputs_batch, margin=margin)))
        # print('predict on iter', ii_counter, loss_batch)

        if loss_batch > max_loss:
            max_loss = loss_batch

        ii_Ndiff += 1
        if ii_Ndiff >= N_diff: # every Ndiff iterations append and reset max_loss
            # print(max_loss)
            list_loss.append(max_loss)
            max_loss = -np.inf
            ii_Ndiff = 0

        ii_counter += 1
        if ii_counter >= iter_time: # after iterating all samples, return mean loss
            return np.mean(list_loss)


def train_embedding_siamese_Ndiff_train_val_routine(list_feature_fold_train,
                                                    labels_fold_train,
                                                    list_feature_fold_val,
                                                    labels_fold_val,
                                                    batch_size,
                                                    input_shape,
                                                    N_diff,
                                                    margin,
                                                    file_path_model,
                                                    file_path_log,
                                                    patience,
                                                    reverse_anchor=False):

    generator_train = generator_triplet_Ndiff(list_feature=list_feature_fold_train,
                                              labels=labels_fold_train,
                                              batch_size=1,
                                              shuffle=True,
                                              reverse_anchor=reverse_anchor,
                                              N_diff=N_diff)

    generator_val = generator_triplet_Ndiff(list_feature=list_feature_fold_val,
                                            labels=labels_fold_val,
                                            batch_size=1,
                                            shuffle=True,
                                            reverse_anchor=reverse_anchor,
                                            N_diff=N_diff)

    embedding_model, triplet_model = embedding_siamese_1_lstm_1_dense(input_shape)

    iter_time_train = len(labels_fold_train)*N_diff/batch_size if not reverse_anchor else len(labels_fold_train)*2*N_diff/batch_size
    iter_time_val = len(labels_fold_val)*N_diff/batch_size if not reverse_anchor else len(labels_fold_val)*2*N_diff/batch_size

    ii_epoch = 0
    ii_patience = 0
    min_val_loss = np.inf
    ii_counter = 0 # batch counter
    ii_Ndiff = 0 # num of diff sample counter
    max_loss = -np.inf # max loss during Ndiff iterations
    input_batch_max_loss = None # input batch with max loss
    for input_batch in generator_train:
        outputs_batch = triplet_model.predict_on_batch(input_batch)
        loss_batch = K.eval(K.mean(triplet_loss(outputs_batch, margin=margin)))
        # print(ii_counter, loss_batch)

        if loss_batch > max_loss:
            max_loss = loss_batch
            input_batch_max_loss = input_batch

        ii_Ndiff += 1
        if ii_Ndiff >= N_diff:
            # print('train on iter', ii_counter, max_loss)
            triplet_model.train_on_batch(input_batch_max_loss, None)
            ii_Ndiff = 0
            max_loss = -np.inf
            input_batch_max_loss = None

        ii_counter += 1
        ii_patience += 1
        if ii_counter >= iter_time_train:
            # train_loss = calculate_loss(triplet_model=triplet_model,
            #                             generator=generator_train,
            #                             iter_time=iter_time_train,
            #                             batch_size=batch_size,
            #                             N_diff=N_diff,
            #                             margin=margin)
            val_loss = calculate_loss(triplet_model=triplet_model,
                                      generator=generator_val,
                                      iter_time=iter_time_val,
                                      batch_size=batch_size,
                                      N_diff=N_diff,
                                      margin=margin)

            writeValLossCsv(file_path_log=file_path_log,
                            ii_epoch=ii_epoch,
                            val_loss=val_loss,
                            train_loss=None)

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                ii_patience = 0
                save_model(triplet_model, filepath=file_path_model)
            elif ii_patience >= patience:
                break

            ii_counter = 0
            ii_epoch += 1


from fit_generator_Ndiff import fit_generator_Ndiff
from feature_generator import generator_triplet_Ndiff_yield_index
from sequence_generator import tripletNdiffYieldIndexSequence


def train_embedding_siamese_Ndiff_train_fit_generator_val_routine(list_feature_fold_train,
                                                                  labels_fold_train,
                                                                  list_feature_fold_val,
                                                                  labels_fold_val,
                                                                  batch_size,
                                                                  input_shape,
                                                                  output_shape,
                                                                  N_diff,
                                                                  margin,
                                                                  file_path_model,
                                                                  file_path_log,
                                                                  patience,
                                                                  verbose,
                                                                  reverse_anchor=False):

    # generator_train = generator_triplet_Ndiff_yield_index(list_feature=list_feature_fold_train,
    #                                                       labels=labels_fold_train,
    #                                                       batch_size=1,
    #                                                       shuffle=True,
    #                                                       reverse_anchor=reverse_anchor,
    #                                                       N_diff=N_diff)
    #
    # generator_val = generator_triplet_Ndiff_yield_index(list_feature=list_feature_fold_val,
    #                                                     labels=labels_fold_val,
    #                                                     batch_size=1,
    #                                                     shuffle=True,
    #                                                     reverse_anchor=reverse_anchor,
    #                                                     N_diff=N_diff)

    generator_train = tripletNdiffYieldIndexSequence(list_feature=list_feature_fold_train,
                                                     labels=labels_fold_train,
                                                     batch_size=batch_size,
                                                     N_diff=N_diff)

    generator_val = tripletNdiffYieldIndexSequence(list_feature=list_feature_fold_val,
                                                   labels=labels_fold_val,
                                                   batch_size=batch_size,
                                                   N_diff=N_diff)

    embedding_model, triplet_model = embedding_siamese_2_lstm_1_dense_model_compile(input_shape=input_shape,
                                                                                    output_shape=output_shape,
                                                                                    margin=margin)

    steps_per_epoch_train = int(np.ceil(len(labels_fold_train) / batch_size))
    steps_per_epcoch_val = int(np.ceil(len(labels_fold_val) / batch_size))

    callbacks = [ModelCheckpoint(file_path_model, monitor='val_loss', verbose=0, save_best_only=True),
                 EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
                 CSVLogger(filename=file_path_log, separator=';')]

    fit_generator_Ndiff(model=triplet_model,
                        generator=generator_train,
                        steps_per_epoch=steps_per_epoch_train,
                        batch_size=batch_size,
                        N_diff=N_diff,
                        margin=margin,
                        epochs=500,
                        verbose=verbose,
                        callbacks=callbacks,
                        validation_data=generator_val,
                        validation_steps=steps_per_epcoch_val,
                        use_multiprocessing=True,
                        shuffle=False)


def train_embedding_siamese_batch(list_feature_fold_train,
                                  labels_fold_train,
                                  list_feature_fold_val,
                                  labels_fold_val,
                                  batch_size,
                                  input_shape,
                                  file_path_model,
                                  filename_log,
                                  patience):

    print("organizing features...")

    # num_same_pairs_train, idx_same_pairs_train = calculate_num_idx_same_pairs(labels_fold_train, 29)
    # num_same_pairs_val, idx_same_pairs_val = calculate_num_idx_same_pairs(labels_fold_val, 29)
    #
    # generator_train = generator_triplet_pairs(list_feature=list_feature_fold_train,
    #                                           labels=labels_fold_train,
    #                                           class_size=np.min(num_same_pairs_train),
    #                                           idx_same_pairs=idx_same_pairs_train,
    #                                           batch_size=1,
    #                                           shuffle=True)
    #
    # generator_val = generator_triplet_pairs(list_feature=list_feature_fold_val,
    #                                         labels=labels_fold_val,
    #                                         class_size=np.min(num_same_pairs_val),
    #                                         idx_same_pairs=idx_same_pairs_val,
    #                                         batch_size=1,
    #                                         shuffle=True)

    generator_train = generator_triplet(list_feature=list_feature_fold_train,
                                        labels=labels_fold_train,
                                        batch_size=1,
                                        shuffle=True,
                                        reverse_anchor=True)

    generator_val = generator_triplet(list_feature=list_feature_fold_val,
                                      labels=labels_fold_val,
                                      batch_size=1,
                                      shuffle=True,
                                      reverse_anchor=True)

    embedding_model, triplet_model = embedding_siamese_1_lstm_1_dense(input_shape)

    callbacks = [ModelCheckpoint(file_path_model, monitor='val_loss', verbose=0, save_best_only=True),
                 EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
                 CSVLogger(filename=filename_log, separator=';')]

    print("start training with validation...")

    triplet_model.fit_generator(generator=generator_train,
                                steps_per_epoch=len(list_feature_fold_train)/batch_size,
                                validation_data=generator_val,
                                validation_steps=len(list_feature_fold_val)/batch_size,
                                callbacks=callbacks,
                                epochs=500,
                                verbose=2)


def train_embedding_siamese_batch_teacher_student(list_feature_fold_train,
                                                  labels_fold_train,
                                                  list_feature_fold_val,
                                                  labels_fold_val,
                                                  batch_size,
                                                  input_shape,
                                                  output_shape,
                                                  margin,
                                                  file_path_model,
                                                  filename_log,
                                                  patience,
                                                  reverse_anchor=False):

    print("organizing features...")

    generator_train = generator_triplet(list_feature=list_feature_fold_train,
                                        labels=labels_fold_train,
                                        batch_size=1,
                                        shuffle=True,
                                        reverse_anchor=reverse_anchor)

    generator_val = generator_triplet(list_feature=list_feature_fold_val,
                                      labels=labels_fold_val,
                                      batch_size=1,
                                      shuffle=True,
                                      reverse_anchor=reverse_anchor)

    embedding_model, triplet_model = embedding_siamese_2_lstm_1_dense_model_compile(input_shape=input_shape,
                                                                                    output_shape=output_shape,
                                                                                    margin=margin)

    callbacks = [ModelCheckpoint(file_path_model, monitor='val_loss', verbose=0, save_best_only=True),
                 EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
                 CSVLogger(filename=filename_log, separator=';')]

    print("start training with validation...")

    triplet_model.fit_generator(generator=generator_train,
                                steps_per_epoch=len(list_feature_fold_train)/batch_size,
                                validation_data=generator_val,
                                validation_steps=len(list_feature_fold_val)/batch_size,
                                callbacks=callbacks,
                                epochs=500,
                                verbose=2)