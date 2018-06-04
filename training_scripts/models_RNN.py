from keras.models import Input
from keras.models import Model
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Reshape
from keras.losses import categorical_crossentropy
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from training_scripts.attention import Attention
from training_scripts.attentionWithContext import AttentionWithContext

import numpy as np

from tensorflow.python.client import device_lib

from training_scripts.data_preparation import writeValLossCsv
from training_scripts.feature_generator import generator_batch_group
from training_scripts.feature_generator import sort_feature_by_seq_length
from training_scripts.feature_generator import batch_grouping


def conv_module(conv, input_shape, input):
    if conv:
        x = Reshape((-1, input_shape[2]) + (1,))(input)
        # x = BatchNormalization()(x)
        x = Conv2D(filters=8, kernel_size=(1, 3), activation="relu")(x)
        # x = BatchNormalization()(x)
        # x = Activation("relu")(x)
        x = Conv2D(filters=8, kernel_size=(1, 3), activation="relu")(x)
        # x = BatchNormalization()(x)
        # x = Activation("relu")(x)
        x = Conv2D(filters=8, kernel_size=(1, 3), activation="relu")(x)
        # x = BatchNormalization()(x)
        # x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(1, 3))(x)

        x = Conv2D(filters=16, kernel_size=(1, 3), activation="relu")(x)
        # x = BatchNormalization()(x)
        # x = Activation("relu")(x)
        x = Conv2D(filters=16, kernel_size=(1, 3), activation="relu")(x)
        # x = BatchNormalization()(x)
        # x = Activation("relu")(x)
        x = Conv2D(filters=16, kernel_size=(1, 3), activation="relu")(x)
        # x = BatchNormalization()(x)
        # x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(1, 3))(x)
        shape = K.int_shape(x)
        x = Reshape((-1, shape[2] * shape[3]))(x)
    else:
        x = input
    return x


def embedding_RNN_1_lstm(input_shape, conv=False, dropout=False):

    device = device_lib.list_local_devices()[0].device_type

    input = Input(batch_shape=input_shape)

    x = conv_module(conv, input_shape, input)

    if device == 'CPU':
        if dropout:
            x = Bidirectional(LSTM(units=32, return_sequences=False, dropout=dropout))(x)
            x = Dropout(dropout)(x)
        else:
            x = Bidirectional(LSTM(units=32, return_sequences=False))(x)
    else:
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=False))(x)

    return x, input


def embedding_RNN_1_lstm_attention(input_shape):
    device = device_lib.list_local_devices()[0].device_type

    input = Input(batch_shape=input_shape)

    if device == 'CPU':
        x = Bidirectional(LSTM(units=32, return_sequences=True))(input)
    else:
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(input)

    x, attention = Attention(return_attention=True)(x)

    return x, input, attention


def embedding_RNN_1_lstm_1_dense(input_shape):

    device = device_lib.list_local_devices()[0].device_type

    input = Input(batch_shape=input_shape)

    if device == 'CPU':
        x = Bidirectional(LSTM(units=32, return_sequences=False))(input)
    else:
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=False))(input)

    x = Dense(units=64, activation='relu')(x)

    x = Dropout(rate=0.5)(x)

    return x, input


def embedding_RNN_2_lstm(input_shape, conv=False, dropout=False):

    device = device_lib.list_local_devices()[0].device_type

    input = Input(batch_shape=input_shape)

    x = conv_module(conv, input_shape, input)

    if device == 'CPU':
        if dropout:
            x = Bidirectional(LSTM(units=32, return_sequences=True, dropout=dropout))(x)
            x = Bidirectional(LSTM(units=32, return_sequences=False, dropout=dropout))(x)
            x = Dropout(dropout)(x)
        else:
            x = Bidirectional(LSTM(units=32, return_sequences=True))(x)
            x = Bidirectional(LSTM(units=32, return_sequences=False))(x)
    else:
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=False))(x)

    return x, input


def embedding_RNN_2_lstm_attention(input_shape, conv, dropout):

    device = device_lib.list_local_devices()[0].device_type

    input = Input(batch_shape=input_shape)

    x = conv_module(conv, input_shape, input)

    if device == 'CPU':
        if dropout:
            x = Bidirectional(LSTM(units=32, return_sequences=True, dropout=dropout))(x)
            x = Bidirectional(LSTM(units=32, return_sequences=True, dropout=dropout))(x)
            x = Dropout(dropout)(x)
        else:
            x = Bidirectional(LSTM(units=32, return_sequences=True))(x)
            x = Bidirectional(LSTM(units=32, return_sequences=True))(x)
    else:
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(input)
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(x)

    x, attention = Attention(return_attention=True)(x)

    return x, input, attention


def embedding_RNN_2_lstm_1_dense(input_shape):

    device = device_lib.list_local_devices()[0].device_type

    input = Input(batch_shape=input_shape)

    if device == 'CPU':
        x = Bidirectional(LSTM(units=32, return_sequences=True))(input)
        x = Bidirectional(LSTM(units=32, return_sequences=False))(x)
    else:
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(input)
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=False))(x)

    x = Dense(units=64, activation='relu')(x)

    x = Dropout(rate=0.5)(x)

    return x, input


def embedding_RNN_2_lstm_2_dense(input_shape):

    device = device_lib.list_local_devices()[0].device_type

    input = Input(batch_shape=input_shape)

    if device == 'CPU':
        x = Bidirectional(LSTM(units=32, return_sequences=True))(input)
        x = Bidirectional(LSTM(units=32, return_sequences=False))(x)
    else:
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(input)
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=False))(x)

    x = Dense(units=64, activation='relu')(x)

    x = Dropout(rate=0.5)(x)

    x = Dense(units=64, activation='relu')(x)

    x = Dropout(rate=0.5)(x)

    return x, input


def embedding_RNN_3_lstm(input_shape):

    device = device_lib.list_local_devices()[0].device_type

    input = Input(batch_shape=input_shape)

    if device == 'CPU':
        x = Bidirectional(LSTM(units=32, return_sequences=True))(input)
        x = Bidirectional(LSTM(units=32, return_sequences=True))(x)
        x = Bidirectional(LSTM(units=32, return_sequences=False))(x)

    else:
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(input)
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=False))(x)

    return x, input


def embedding_RNN_3_lstm_1_dense(input_shape):

    device = device_lib.list_local_devices()[0].device_type

    input = Input(batch_shape=input_shape)

    if device == 'CPU':
        x = Bidirectional(LSTM(units=32, return_sequences=True))(input)
        x = Bidirectional(LSTM(units=32, return_sequences=True))(x)
        x = Bidirectional(LSTM(units=32, return_sequences=False))(x)
    else:
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(input)
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=False))(x)

    x = Dense(units=64, activation='relu')(x)

    x = Dropout(rate=0.5)(x)

    return x, input


def embedding_RNN_3_lstm_2_dense(input_shape):

    device = device_lib.list_local_devices()[0].device_type

    input = Input(batch_shape=input_shape)

    if device == 'CPU':
        x = Bidirectional(LSTM(units=32, return_sequences=True))(input)
        x = Bidirectional(LSTM(units=32, return_sequences=True))(x)
        x = Bidirectional(LSTM(units=32, return_sequences=False))(x)
    else:
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(input)
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=False))(x)

    x = Dense(units=64, activation='relu')(x)

    x = Dropout(rate=0.5)(x)

    x = Dense(units=64, activation='relu')(x)

    x = Dropout(rate=0.5)(x)

    return x, input


def embedding_RNN_3_lstm_3_dense(input_shape):

    device = device_lib.list_local_devices()[0].device_type

    input = Input(batch_shape=input_shape)

    if device == 'CPU':
        x = Bidirectional(LSTM(units=32, return_sequences=True))(input)
        x = Bidirectional(LSTM(units=32, return_sequences=True))(x)
        x = Bidirectional(LSTM(units=32, return_sequences=False))(x)
    else:
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(input)
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(units=32, return_sequences=False))(x)

    x = Dense(units=64, activation='relu')(x)

    x = Dropout(rate=0.5)(x)

    x = Dense(units=64, activation='relu')(x)

    x = Dropout(rate=0.5)(x)

    x = Dense(units=64, activation='relu')(x)

    x = Dropout(rate=0.5)(x)

    return x, input


def evaluate_model(model, X, y, scaler):

    y_pred = np.zeros_like(y)
    for ii in range(len(X)):
        X_sample = np.expand_dims(scaler.transform(X[ii]), axis=0)
        y_pred[ii] = model.predict_on_batch(X_sample)

    print(y.shape, y_pred.shape)
    y = K.variable(y)
    y_pred = K.variable(y_pred)

    loss = K.eval(categorical_crossentropy(y, y_pred))

    return np.mean(loss)


def shuffleFeaturesLabelsInUnison(features, labels):
    p=np.random.permutation(len(features))
    features = [features[ii] for ii in p]
    return features, labels[p]


def train_embedding_RNN(X_train,
                        X_val,
                        y_train,
                        y_val,
                        scaler,
                        input_shape,
                        file_path_model,
                        file_path_log):
    model = embedding_RNN_1_lstm(input_shape=input_shape)

    nb_epochs = 500
    best_val_loss = 1.0  # initialize the val_loss
    counter = 0
    patience = 15  # early stopping patience

    for ii_epoch in range(nb_epochs):
        for ii in range(len(X_train)):
            X_train_sample = np.expand_dims(scaler.transform(X_train[ii]), axis=0)
            y_train_sample = np.expand_dims(y_train[ii], axis=0)
            # print(X_train_sample)
            results = model.train_on_batch(X_train_sample, y_train_sample)

        train_loss = evaluate_model(model=model, X=X_train, y=y_train, scaler=scaler)
        val_loss = evaluate_model(model=model, X=X_val, y=y_val, scaler=scaler)

        print(train_loss, val_loss)

        # save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            model.save_weights(file_path_model)
        else:
            counter += 1

        # write validation loss to csv
        writeValLossCsv(file_path_log=file_path_log,
                        ii_epoch=ii_epoch,
                        val_loss=val_loss,
                        train_loss=train_loss)

        # early stopping
        if counter >= patience:
            break

        X_train, y_train = shuffleFeaturesLabelsInUnison(features=X_train, labels=y_train)


def model_select(config, input_shape, conv, dropout):
    if config[0] == 1 and config[1] == 0:
        x = embedding_RNN_1_lstm(input_shape=input_shape, conv=conv, dropout=dropout)
    elif config[0] == 1 and config[1] == 1:
        x = embedding_RNN_1_lstm_1_dense(input_shape=input_shape)
    elif config[0] == 2 and config[1] == 0:
        x = embedding_RNN_2_lstm(input_shape=input_shape, conv=conv, dropout=dropout)
    elif config[0] == 2 and config[1] == 1:
        x = embedding_RNN_2_lstm_1_dense(input_shape=input_shape)
    elif config[0] == 2 and config[1] == 2:
        x = embedding_RNN_2_lstm_2_dense(input_shape=input_shape)
    elif config[0] == 3 and config[1] == 0:
        x = embedding_RNN_3_lstm(input_shape=input_shape)
    elif config[0] == 3 and config[1] == 1:
        x = embedding_RNN_3_lstm_1_dense(input_shape=input_shape)
    elif config[0] == 3 and config[1] == 2:
        x = embedding_RNN_3_lstm_2_dense(input_shape=input_shape)
    elif config[0] == 3 and config[1] == 3:
        x = embedding_RNN_3_lstm_3_dense(input_shape=input_shape)
    else:
        raise ValueError

    return x


def model_select_attention(config, input_shape, conv, dropout):
    if config[0] == 1 and config[1] == 0:
        x = embedding_RNN_1_lstm_attention(input_shape=input_shape)
    elif config[0] == 2 and config[1] == 0:
        x = embedding_RNN_2_lstm_attention(input_shape=input_shape, conv=conv, dropout=dropout)
    else:
        raise ValueError
    return x


def train_embedding_RNN_batch(list_feature_fold_train,
                              labels_fold_train,
                              list_feature_fold_val,
                              labels_fold_val,
                              batch_size,
                              input_shape,
                              output_shape,
                              file_path_model,
                              filename_log,
                              patience,
                              config,
                              attention,
                              dense,
                              conv,
                              dropout):

    print("organizing features...")

    list_feature_sorted_train, labels_sorted_train, iter_times_train = \
        sort_feature_by_seq_length(list_feature=list_feature_fold_train,
                                   labels=labels_fold_train,
                                   batch_size=batch_size)

    list_feature_sorted_val, labels_sorted_val, iter_times_val = \
        sort_feature_by_seq_length(list_feature=list_feature_fold_val,
                                   labels=labels_fold_val,
                                   batch_size=batch_size)

    list_X_batch_train, list_y_batch_train = batch_grouping(list_feature_sorted=list_feature_sorted_train,
                                                            labels_sorted=labels_sorted_train,
                                                            batch_size=batch_size,
                                                            iter_times=iter_times_train)

    list_X_batch_val, list_y_batch_val = batch_grouping(list_feature_sorted=list_feature_sorted_val,
                                                        labels_sorted=labels_sorted_val,
                                                        batch_size=batch_size,
                                                        iter_times=iter_times_val)

    generator_train = generator_batch_group(list_X_batch=list_X_batch_train,
                                            list_y_batch=list_y_batch_train,
                                            iter_times=iter_times_train)

    generator_val = generator_batch_group(list_X_batch=list_X_batch_val,
                                          list_y_batch=list_y_batch_val,
                                          iter_times=iter_times_val)

    if attention:
        x, input, _ = model_select_attention(config=config, input_shape=input_shape, conv=conv, dropout=dropout)
    else:
        x, input = model_select(config=config, input_shape=input_shape, conv=conv, dropout=dropout)

    if dense:
        x = Dense(32)(x)

    outputs = Dense(output_shape, activation='softmax')(x)
    model = Model(inputs=input, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    callbacks = [ModelCheckpoint(file_path_model, monitor='val_loss', verbose=0, save_best_only=True),
                 EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
                 CSVLogger(filename=filename_log, separator=';')]

    print("start training with validation...")

    model.fit_generator(generator=generator_train,
                        steps_per_epoch=iter_times_train,
                        validation_data=generator_val,
                        validation_steps=iter_times_val,
                        callbacks=callbacks,
                        epochs=500,
                        verbose=2)


def train_embedding_RNN_batch_MTL(list_feature_fold_train,
                                  labels_fold_train,
                                  list_feature_fold_val,
                                  labels_fold_val,
                                  batch_size,
                                  input_shape,
                                  output_shape,
                                  file_path_model,
                                  filename_log,
                                  patience,
                                  config,
                                  attention,
                                  dense=False,
                                  conv=False,
                                  dropout=False):

    print("organizing features...")

    list_feature_sorted_train, labels_sorted_train, iter_times_train = \
        sort_feature_by_seq_length(list_feature=list_feature_fold_train,
                                   labels=labels_fold_train,
                                   batch_size=batch_size)

    list_feature_sorted_val, labels_sorted_val, iter_times_val = \
        sort_feature_by_seq_length(list_feature=list_feature_fold_val,
                                   labels=labels_fold_val,
                                   batch_size=batch_size)

    list_X_batch_train, list_y_batch_train = batch_grouping(list_feature_sorted=list_feature_sorted_train,
                                                            labels_sorted=labels_sorted_train,
                                                            batch_size=batch_size,
                                                            iter_times=iter_times_train)

    list_X_batch_val, list_y_batch_val = batch_grouping(list_feature_sorted=list_feature_sorted_val,
                                                        labels_sorted=labels_sorted_val,
                                                        batch_size=batch_size,
                                                        iter_times=iter_times_val)

    generator_train = generator_batch_group(list_X_batch=list_X_batch_train,
                                            list_y_batch=list_y_batch_train,
                                            iter_times=iter_times_train)

    generator_val = generator_batch_group(list_X_batch=list_X_batch_val,
                                          list_y_batch=list_y_batch_val,
                                          iter_times=iter_times_val)

    if attention:
        x, input, _ = model_select_attention(config=config, input_shape=input_shape)
    else:
        x, input = model_select(config=config, input_shape=input_shape, conv=conv, dropout=dropout)

    pronun_out = Dense(output_shape[0], activation='softmax', name='pronunciation')(x)

    if dense:
        x = Dense(units=32)(x)

    profess_out = Dense(output_shape[1], activation='softmax', name='professionality')(x)

    model = Model(inputs=input, outputs=[pronun_out, profess_out])

    # optimizer = SGD(lr=0.05, momentum=0.45, decay=0.0, nesterov=False)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  loss_weights=[0.5, 0.5])

    model.summary()

    callbacks = [ModelCheckpoint(file_path_model, monitor='val_loss', verbose=0, save_best_only=True),
                 EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
                 CSVLogger(filename=filename_log, separator=';')]

    print("start training with validation...")

    model.fit_generator(generator=generator_train,
                        steps_per_epoch=iter_times_train,
                        validation_data=generator_val,
                        validation_steps=iter_times_val,
                        callbacks=callbacks,
                        epochs=500,
                        verbose=2)