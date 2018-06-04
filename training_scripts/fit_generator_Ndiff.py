import numpy as np
import warnings
from keras import backend as K
from keras import callbacks as cbks
from keras.utils.data_utils import Sequence
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.data_utils import GeneratorEnqueuer

from training_scripts.losses import triplet_loss
from training_scripts.losses import triplet_loss_no_mean


def get_maximum_length(batch_size, generator_output, index):
    """
    retrieve the maximum sequence length of each batch
    :param batch_size:
    :param generator_output: [[N_diff samples], ...], N_diff samples is [[anchor_0, same_0, diff_0], ...]
    :param index: array
    :return:
    """
    # get the maximum sequence length
    len_anchor_max, len_same_max, len_diff_max = 0, 0, 0
    for ii_sample in range(batch_size):
        gen_out_sample = generator_output[ii_sample][index[ii_sample]]
        if gen_out_sample[0].shape[1] > len_anchor_max: len_anchor_max = gen_out_sample[0].shape[1]
        if gen_out_sample[1].shape[1] > len_same_max: len_same_max = gen_out_sample[1].shape[1]
        if gen_out_sample[2].shape[1] > len_diff_max: len_diff_max = gen_out_sample[2].shape[1]

    return len_anchor_max, len_same_max, len_diff_max


def make_same_length_batch(batch_size, len_anchor_max, len_same_max, len_diff_max, generator_output, index):
    """
    make the input matrix with the maximum length
    :param batch_size:
    :param len_anchor_max:
    :param len_same_max:
    :param len_diff_max:
    :param generator_output: [[N_diff samples], ...], N_diff samples is [[anchor_0, same_0, diff_0], ...]
    :param index: array
    :return:
    """
    # padding
    input_anchor = np.zeros((batch_size, len_anchor_max, 80), dtype=np.float32)
    input_same = np.zeros((batch_size, len_same_max, 80), dtype=np.float32)
    input_diff = np.zeros((batch_size, len_diff_max, 80), dtype=np.float32)
    for ii_sample in range(batch_size):
        gen_out_sample = generator_output[ii_sample][index[ii_sample]]
        input_anchor[ii_sample, :gen_out_sample[0].shape[1], :] = gen_out_sample[0]
        input_same[ii_sample, :gen_out_sample[1].shape[1], :] = gen_out_sample[1]
        input_diff[ii_sample, :gen_out_sample[2].shape[1], :] = gen_out_sample[2]

    return input_anchor, input_same, input_diff


def fit_generator_Ndiff(model,
                        generator,
                        steps_per_epoch=None,
                        batch_size=1,
                        N_diff=5,
                        margin=0.5,
                        epochs=1,
                        verbose=1,
                        callbacks=None,
                        validation_data=None,
                        validation_steps=None,
                        class_weight=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        shuffle=True,
                        initial_epoch=0):
    """Trains the model on data yielded batch-by-batch by a Python generator.
    The generator is run in parallel to the model, for efficiency.
    For instance, this allows you to do real-time data augmentation
    on images on CPU in parallel to training your model on GPU.
    The use of `keras.utils.Sequence` guarantees the ordering
    and guarantees the single use of every input per epoch when
    using `use_multiprocessing=True`.
    # Arguments
        generator: A generator or an instance of `Sequence`
            (`keras.utils.Sequence`) object in order to avoid
            duplicate data when using multiprocessing.
            The output of the generator must be either
            - a tuple `(inputs, targets)`
            - a tuple `(inputs, targets, sample_weights)`.
            This tuple (a single output of the generator) makes a single
            batch. Therefore, all arrays in this tuple must have the same
            length (equal to the size of this batch). Different batches
            may have different sizes. For example, the last batch of the
            epoch is commonly smaller than the others, if the size of the
            dataset is not divisible by the batch size.
            The generator is expected to loop over its data
            indefinitely. An epoch finishes when `steps_per_epoch`
            batches have been seen by the model.
        steps_per_epoch: Integer.
            Total number of steps (batches of samples)
            to yield from `generator` before declaring one epoch
            finished and starting the next epoch. It should typically
            be equal to the number of samples of your dataset
            divided by the batch size.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire data provided,
            as defined by `steps_per_epoch`.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        verbose: Integer. 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See [callbacks](/callbacks).
        validation_data: This can be either
            - a generator for the validation data
            - tuple `(x_val, y_val)`
            - tuple `(x_val, y_val, val_sample_weights)`
            on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data.
        validation_steps: Only relevant if `validation_data`
            is a generator. Total number of steps (batches of samples)
            to yield from `validation_data` generator before stopping.
            Optional for `Sequence`: if unspecified, will use
            the `len(validation_data)` as a number of steps.
        class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only).
            This can be useful to tell the model to
            "pay more attention" to samples from
            an under-represented class.
        max_queue_size: Integer. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Maximum number of processes to spin up
            when using process based threading.
            If unspecified, `workers` will default to 1. If 0, will
            execute the generator on the main thread.
        use_multiprocessing: Boolean. If True, use process based threading.
            If unspecified, `use_multiprocessing` will default to False.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
        shuffle: Boolean. Whether to shuffle the training data
            in batch-sized chunks before each epoch.
            Only used with instances of `Sequence` (`keras.utils.Sequence`).
        initial_epoch: Integer.
            Epoch at which to start training
            (useful for resuming a previous training run).
    # Returns
        A `History` object. Its `History.history` attribute is
        a record of training loss values and metrics values
        at successive epochs, as well as validation loss values
        and validation metrics values (if applicable).
    # Example
    ```python
        def generate_arrays_from_file(path):
            while 1:
                with open(path) as f:
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x1, x2, y = process_line(line)
                        yield ({'input_1': x1, 'input_2': x2}, {'output': y})
        model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                            steps_per_epoch=10000, epochs=10)
    ```
    # Raises
        ValueError: In case the generator yields
            data in an invalid format.
    """
    wait_time = 0.01  # in seconds
    epoch = initial_epoch

    do_validation = bool(validation_data)
    # self._make_train_function()
    # if do_validation:
    #     self._make_test_function()

    is_sequence = isinstance(generator, Sequence)
    # do_validation = True if is_sequence else False

    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    if steps_per_epoch is None:
        if is_sequence:
            steps_per_epoch = len(generator)
        else:
            raise ValueError('`steps_per_epoch=None` is only valid for a'
                             ' generator based on the `keras.utils.Sequence`'
                             ' class. Please specify `steps_per_epoch` or use'
                             ' the `keras.utils.Sequence` class.')

    # python 2 has 'next', 3 has '__next__'
    # avoid any explicit version checks
    val_gen = (hasattr(validation_data, 'next') or
               hasattr(validation_data, '__next__') or
               isinstance(validation_data, Sequence))
    if (val_gen and not isinstance(validation_data, Sequence) and
            not validation_steps):
        raise ValueError('`validation_steps=None` is only valid for a'
                         ' generator based on the `keras.utils.Sequence`'
                         ' class. Please specify `validation_steps` or use'
                         ' the `keras.utils.Sequence` class.')

    # Prepare display labels.
    out_labels = model._get_deduped_metrics_names()
    callback_metrics = out_labels + ['val_' + n for n in out_labels]

    # prepare callbacks
    history = cbks.History()
    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [history]
    if verbose:
        callbacks += [cbks.ProgbarLogger(count_mode='steps')]
    callbacks = cbks.CallbackList(callbacks)

    # # it's possible to callback a different model than self:
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model
    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()

    enqueuer = None
    val_enqueuer = None

    try:
        if do_validation:
            if val_gen:
                if workers > 0:
                    if isinstance(validation_data, Sequence):
                        val_enqueuer = OrderedEnqueuer(
                            validation_data,
                            use_multiprocessing=use_multiprocessing)
                        if validation_steps is None:
                            validation_steps = len(validation_data)
                    else:
                        val_enqueuer = GeneratorEnqueuer(
                            validation_data,
                            use_multiprocessing=use_multiprocessing,
                            wait_time=wait_time)
                    val_enqueuer.start(workers=workers, max_queue_size=max_queue_size)
                    validation_generator = val_enqueuer.get()
                else:
                    validation_generator = validation_data
            else:
                pass
                # if len(validation_data) == 2:
                #     val_x, val_y = validation_data
                #     val_sample_weights = None
                # elif len(validation_data) == 3:
                #     val_x, val_y, val_sample_weights = validation_data
                # else:
                #     raise ValueError('`validation_data` should be a tuple '
                #                      '`(val_x, val_y, val_sample_weight)` '
                #                      'or `(val_x, val_y)`. Found: ' +
                #                      str(validation_data))
                # val_x, val_y, val_sample_weights = _standardize_user_data(
                #     val_x, val_y, val_sample_weight)
                # val_data = val_x + val_y + val_sample_weights
                # if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                #     val_data += [0.]
                # for cbk in callbacks:
                #     cbk.validation_data = val_data

        if workers > 0:
            if is_sequence:
                enqueuer = OrderedEnqueuer(generator,
                                           use_multiprocessing=use_multiprocessing,
                                           shuffle=shuffle)
            else:
                enqueuer = GeneratorEnqueuer(generator,
                                             use_multiprocessing=use_multiprocessing,
                                             wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            output_generator = generator

        callback_model.stop_training = False
        # Construct epoch logs.
        epoch_logs = {}
        while epoch < epochs:
            callbacks.on_epoch_begin(epoch)
            steps_done = 0
            batch_index = 0
            while steps_done < steps_per_epoch:
                generator_output = next(output_generator)

                if not hasattr(generator_output, '__len__'):
                    raise ValueError('Output of generator should be '
                                     'batch_size lists ' +
                                     str(generator_output))

                if len(generator_output) == batch_size:
                    # ii_ndiff: the index of the negative sample
                    gen_out = generator_output
                    sample_weight = None
                else:
                    raise ValueError('Output of generator should be '
                                     'batch_size lists ' +
                                     str(generator_output))

                # build batch logs
                batch_logs = {}
                # if isinstance(x, list):
                #     batch_size = x[0].shape[0]
                # elif isinstance(x, dict):
                #     batch_size = list(x.values())[0].shape[0]
                # else:
                #     batch_size = x.shape[0]
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_batch_begin(batch_index, batch_logs)

                # aggregate the losses by inner index n_diff
                loss_mat = np.zeros((batch_size, N_diff))
                for ii_ndiff in range(N_diff):

                    # get the maximum sequence length
                    # use ii_ndiff sample in N_diff
                    len_anchor_max, len_same_max, len_diff_max = \
                        get_maximum_length(batch_size=batch_size,
                                           generator_output=gen_out,
                                           index=[ii_ndiff]*batch_size)

                    # print(len_anchor_max, len_same_max, len_diff_max)
                    # pad the samples to the same length
                    # organize the input for the prediction
                    input_anchor, input_same, input_diff = \
                        make_same_length_batch(batch_size=batch_size,
                                               len_anchor_max=len_anchor_max,
                                               len_same_max=len_same_max,
                                               len_diff_max=len_diff_max,
                                               generator_output=gen_out,
                                               index=[ii_ndiff]*batch_size)

                    output_batch_pred = model.predict_on_batch([input_anchor, input_same, input_diff])

                    loss = K.eval(triplet_loss_no_mean(output_batch_pred, margin))
                    loss_mat[:, ii_ndiff] = loss

                # this the index of the input which has the maximum loss for each N_diff pairs
                # index_max_loss dim: [batch_size, 1]
                index_max_loss = np.argmax(loss_mat, axis=-1)

                len_anchor_max, len_same_max, len_diff_max = get_maximum_length(batch_size=batch_size,
                                                                                generator_output=gen_out,
                                                                                index=index_max_loss)

                # input_anchor, input_same, input_diff dim: [batch_size, length, feature_dim]
                input_anchor, input_same, input_diff = \
                    make_same_length_batch(batch_size=batch_size,
                                           len_anchor_max=len_anchor_max,
                                           len_same_max=len_same_max,
                                           len_diff_max=len_diff_max,
                                           generator_output=gen_out,
                                           index=index_max_loss)

                outs = model.train_on_batch([input_anchor, input_same, input_diff], None,
                                            sample_weight=sample_weight,
                                            class_weight=class_weight)

                if not isinstance(outs, list):
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                batch_index += 1
                steps_done += 1

                # Epoch finished.
                if steps_done >= steps_per_epoch and do_validation:
                    if val_gen:
                        val_outs = evaluate_generator(model=model,
                                                      generator=validation_generator,
                                                      steps=validation_steps,
                                                      batch_size=batch_size,
                                                      margin=margin,
                                                      N_diff=N_diff,
                                                      workers=0)
                    else:
                        pass
                        # # No need for try/except because
                        # # data has already been validated.
                        # val_outs = model.evaluate(
                        #     val_x, val_y,
                        #     batch_size=batch_size,
                        #     sample_weight=val_sample_weights,
                        #     verbose=0)
                    if not isinstance(val_outs, list):
                        val_outs = [val_outs]
                    # Same labels assumed.
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o

                if callback_model.stop_training:
                    break

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
            if callback_model.stop_training:
                break

    finally:
        try:
            if enqueuer is not None:
                enqueuer.stop()
        finally:
            if val_enqueuer is not None:
                val_enqueuer.stop()

    callbacks.on_train_end()
    return history


def evaluate_generator(model,
                       generator,
                       steps=None,
                       batch_size=1,
                       margin=0.5,
                       N_diff=5,
                       max_queue_size=10,
                       workers=1,
                       use_multiprocessing=False):
    """Evaluates the model on a data generator.
    The generator should return the same kind of data
    as accepted by `test_on_batch`.
    # Arguments
        generator: Generator yielding tuples (inputs, targets)
            or (inputs, targets, sample_weights)
            or an instance of Sequence (keras.utils.Sequence)
            object in order to avoid duplicate data
            when using multiprocessing.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
            Optional for `Sequence`: if unspecified, will use
            the `len(generator)` as a number of steps.
        max_queue_size: maximum size for the generator queue
        workers: Integer. Maximum number of processes to spin up
            when using process based threading.
            If unspecified, `workers` will default to 1. If 0, will
            execute the generator on the main thread.
        use_multiprocessing: if True, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non picklable arguments to the generator
            as they can't be passed
            easily to children processes.
    # Returns
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    # Raises
        ValueError: In case the generator yields
            data in an invalid format.
    """
    # self._make_test_function()

    steps_done = 0
    wait_time = 0.01
    all_outs = []
    batch_sizes = []
    is_sequence = isinstance(generator, Sequence)
    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    if steps is None:
        if is_sequence:
            steps = len(generator)
        else:
            raise ValueError('`steps=None` is only valid for a generator'
                             ' based on the `keras.utils.Sequence` class.'
                             ' Please specify `steps` or use the'
                             ' `keras.utils.Sequence` class.')
    enqueuer = None

    try:
        if workers > 0:
            if is_sequence:
                enqueuer = OrderedEnqueuer(generator,
                                           use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(generator,
                                             use_multiprocessing=use_multiprocessing,
                                             wait_time=wait_time)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            output_generator = generator

        while steps_done < steps:
            generator_output = next(output_generator)
            if not hasattr(generator_output, '__len__'):
                raise ValueError('Output of generator should be a tuple '
                                 '(x, y, z, ii_ndiff) ' +
                                 str(generator_output))
            if len(generator_output) == batch_size:
                gen_out = generator_output
                sample_weight = None
            else:
                raise ValueError('Output of generator should be a tuple '
                                 '(x, y, z, ii_ndiff) ' +
                                 str(generator_output))

            loss_mat = np.zeros((batch_size, N_diff))
            for ii_ndiff in range(N_diff):
                # get the maximum sequence length
                len_anchor_max, len_same_max, len_diff_max = \
                    get_maximum_length(batch_size=batch_size,
                                       generator_output=gen_out,
                                       index=[ii_ndiff] * batch_size)

                # print(len_anchor_max, len_same_max, len_diff_max)
                # organize the input for the prediction
                input_anchor, input_same, input_diff = \
                    make_same_length_batch(batch_size=batch_size,
                                           len_anchor_max=len_anchor_max,
                                           len_same_max=len_same_max,
                                           len_diff_max=len_diff_max,
                                           generator_output=gen_out,
                                           index=[ii_ndiff] * batch_size)

                output_batch_pred = model.predict_on_batch([input_anchor, input_same, input_diff])

                loss = K.eval(triplet_loss_no_mean(output_batch_pred, margin))
                loss_mat[:, ii_ndiff] = loss

            outs = np.mean(np.max(loss_mat, axis=-1))

            # if isinstance(x, list):
            #     batch_size = x[0].shape[0]
            # elif isinstance(x, dict):
            #     batch_size = list(x.values())[0].shape[0]
            # else:
            #     batch_size = x.shape[0]
            # if batch_size == 0:
            #     raise ValueError('Received an empty batch. '
            #                      'Batches should at least contain one item.')
            all_outs.append(outs)

            steps_done += 1
            batch_sizes.append(batch_size)

    finally:
        if enqueuer is not None:
            enqueuer.stop()

    if not isinstance(outs, list):
        return np.average(np.asarray(all_outs),
                          weights=batch_sizes)
    else:
        averages = []
        for i in range(len(outs)):
            averages.append(np.average([out[i] for out in all_outs],
                                       weights=batch_sizes))
        return averages