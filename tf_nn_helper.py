"""
tf_nn_helper.py - helper functions for creation, training, evaluation & prediction of Tensorflow models.
This file include utility functions that preclude writing of boilerplate code to train/evaluate a Tensorflow
model. It also provides a utility function to report training progress using a Keras-like progressbar.

NOTE: Some of the functions begin with __ (double underscore) - these should be treated as internal/private
functions, which are called from the rest of the functions provided here (i.e. functions whose name does not
begin with __). Following are the public functions provided:
    
  Tensorflow model creation helper function:
    - conv2d(x, W, b, strides=1, padding='SAME', activation='relu', batch_normalize=False, dilations=[1,1,
    1,1])
        creates a conv2d layer, applying the activation function and optionally batch normalization,
        if specified
    - maxpool2d()
    - dense()
    - flatten()
    - TODO: add functions documentation...
  Tensorflow model training/evaluation/prediction helper functions:
    - train_model()
    - evaluate_model()
    - predict()
    - fit_generator()
    - evaluate_generator()
    - predict_generator()

@author: Manish Bhobe
Use this code at your own risk. I am not responsible if your computer explodes of GPU gets fried :P

Usage: 
    - Copy file to same folder as your iPython notebook(s) or Python module(s)
    - In the imports section add the following line
        from tf_nn_helper import *
"""

# imports & tweaks
import numpy as np
import sys, time, os
import tensorflow as tf

import kr_helper_funcs as kru

seed = 101
np.random.seed(seed)
tf.set_random_seed(seed)

# ----------------------------------------------------------------------------------------
# these calls map directly to functions in kr_helper_funcs.py
# ----------------------------------------------------------------------------------------
def progbar_msg(curr_tick, max_tick, head_msg, tail_msg, final=False):
    kru.progbar_msg(curr_tick, max_tick, head_msg, tail_msg, final=False)

def show_plots(history):
    kru.show_plots(history)

def time_taken_as_str(start_time, end_time):
    return kru.time_taken_as_str(start_time, end_time)

# Global activation functions map
tf_activation_fxns_map = {
    # covering most commonly used activation functions
    # please add to this as appropriate
    'relu'      : tf.nn.relu, 'elu': tf.nn.elu, 'sigmoid': tf.nn.sigmoid, 'tanh': tf.nn.tanh,
    'leaky_relu': tf.nn.leaky_relu, 'softmax': tf.nn.softmax
}

# -----------------------------------------------------------------------------------------------
# Tensorflow model creation helper functions
# -----------------------------------------------------------------------------------------------
def conv2d(x, W, b, strides=1, padding='SAME', activation='relu', batch_normalize=False,
           dilations=[1, 1, 1, 1]):
    """ creates a 2d convolution layer with the following parameters:
      x - a 4D tensor of shape [num_samples, height, width, num_channels]
      W - a 4D weight tensor for this layer of shape [kernel_size, kernel_size, channels_in, channels_out]
      b - a 1D bias tensor for this layer of shape [channels_out]
      strides - the stride of the sliding window (default=1)
      activation - activation function (default='relu' - one of 'relu','elu','sigmoid','tanh',
      'leaky_relu'
    """
    # some assertions
    assert strides >= 1
    assert padding.upper() in ['SAME', 'VALID']
    assert activation in tf_activation_fxns_map.keys()

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding, dilations=dilations)
    x = tf.nn.bias_add(x, b)
    if batch_normalize:
        x = tf.layers.batch_normalization(x)  # all other params are default
    return tf_activation_fxns_map[activation](x)

def maxpool2d(x, pool_size=2, padding='VALID'):
    """ creates a max-pooling layer, with given pool_size (defaults to 2)"""
    assert padding.upper() in ['SAME', 'VALID']
    return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, pool_size, pool_size, 1],
                          padding=padding)

def flatten(layer, weights_fc1):
    """ flattens out a layer before connecting to a dense layers 
        weights_fc1 = first fully connected layer
    """
    flt = tf.reshape(layer, [-1, weights_fc1.get_shape().as_list()[0]])
    return flt

def dense(x, W, b, activation='relu', batch_normalize=False):
    """ creates a dense (or fully connected) layer 
    @params:
        x - previous or input layer tensor
        W - weights Variable for this layer
        b - bias Variable for this layer
    @returns:
        tf.nn.relu(tf.matmul(x,W) + b)
    """
    assert activation in tf_activation_fxns_map.keys()

    dl = tf.add(tf.matmul(x, W), b)
    if batch_normalize:
        dl = tf.layers.batch_normalization(dl)  # all other params are default

    return tf_activation_fxns_map[activation](dl)

def add_name_to_tensor(t, name):
    """ adds a name to a tensor & returns the same tensor """
    return tf.identity(t, name=name)

# -----------------------------------------------------------------------------------------------
# Tensorflow model training helper functions
# -----------------------------------------------------------------------------------------------

def __update_progbar(batch_num=None, num_batches=25, train_cost=None, train_acc=None, val_test_cost=None,
                     val_test_acc=None, phase='train', final=False, eta=None):
    phase = phase.lower()
    if phase not in ['train', 'valid', 'eval', 'done']:
        raise ValueError('Unknown value for phase param - %s' % phase)

    phase_lookup = {
        'train': 'Training', 'valid': 'Validating', 'eval': 'Evaluating', 'done': 'Completed!'
    }

    # -----------------------------------------------------------------------------------------
    # this is a helper function that displays a Keras-like progress bar as the Tensorflow model 
    # does a batch training or testing across epochs
    # NOTE: In hindsight, this is not a very extensible function. Probably a better approach
    # would be to create an extensible class which can be extended to provide progress updates
    # on_epoch_end, on_batch_end etc. For now, it works, so I am not writing this.
    # Maybe next time :)
    # ---------------------------------------------------------------------------
    progbar_num_ticks = 30

    def get_prog_ticks(batch_num):
        # num_batchs <=> progbar_num_ticks 
        prog_tick = (batch_num * progbar_num_ticks) // num_batches
        prog_tick = (1 if prog_tick <= 0 else prog_tick)
        bal_tick = (progbar_num_ticks - prog_tick)
        return prog_tick, bal_tick

    def eta_str(eta):
        SECS_PER_MIN = 60
        SECS_PER_HR = 60 * SECS_PER_MIN

        hrs_elapsed, secs_elapsed = divmod(eta, SECS_PER_HR)
        mins_elapsed, secs_elapsed = divmod(eta, SECS_PER_MIN)

        if hrs_elapsed > 0:
            return 'ETA: %dh %dm %ds - ' % (hrs_elapsed, mins_elapsed, secs_elapsed)
        elif mins_elapsed > 0:
            return 'ETA: %dm %ds - ' % (mins_elapsed, secs_elapsed)
        elif secs_elapsed > 1:
            return 'ETA: %ds - ' % (secs_elapsed)
        else:
            return 'ETA: <1s - '

    len_num_batches = len(str(num_batches))

    if batch_num is None:
        # initial display only
        prog_bar = '  %s will commence shortly. Please wait...' % phase_lookup[phase]
    else:
        prog_tick, bal_tick = get_prog_ticks(batch_num)

        if not final:
            # Training or Validating message
            prog_bar = '  %s : Batch (%*d/%*d) [%s%s%s] -> %sloss: %.4f - acc: %.4f%s' % (
                phase_lookup[phase], len_num_batches, batch_num, len_num_batches, num_batches,
                ('=' * (prog_tick - 1)), '>', ('.' * (bal_tick)), (eta_str(eta) if eta is not None else ''),
                train_cost, train_acc, ' ' * 10)
        else:  # final display
            # erase previous line
            prog_bar = ' ' * 85
            print('\r%s' % prog_bar, end='', flush=True)
            # it may not contain validation data
            if ((val_test_cost is None) or (val_test_acc is None)):
                prog_bar = '  %s Batch (%*d/%*d) [%s] -> loss: %.4f - acc: %.4f%s\n' % (
                    (phase_lookup[phase] + ' :' if phase == 'eval' else ''), len_num_batches, batch_num,
                    len_num_batches, num_batches, ('=' * (progbar_num_ticks)), train_cost, train_acc,
                    ' ' * 40)
            else:
                prog_bar = '  %s Batch (%*d/%*d) [%s] -> loss: %.4f - acc: %.4f - val_loss: %.4f ' \
                           '- val_acc: %.4f%s\n' % (
                               (phase_lookup[phase] + ' :' if phase == 'eval' else ''), len_num_batches,
                               batch_num, len_num_batches, num_batches, ('=' * (progbar_num_ticks)),
                               train_cost, train_acc, val_test_cost, val_test_acc, ' ' * 20)

    # @see: https://github.com/spyder-ide/spyder/issues/3437, which suggests printing
    # in the following way, so that it works on all consoles - but does not work on Terminal :(
    print('\r%s' % prog_bar, end='', flush=True)  # print(prog_bar, end='\r', flush=True)

def __batch_calc_loss_acc(sess, model, data, labels, feed_dict=None, batch_size=32, do_training=False,
                          show_progbar=True, phase='train', final=False):
    # -------------------------------------------------------------------------------------------
    # batch calculates loss & accuracy using loss & accuracy 'attributes' of the model dict
    # batch calculations preclude OOM errors in Tensorflow - does not avoid them, but you could
    # tweak the batch size to almost eliminate them
    # NOTE: pass only a fitted Karas ImageDataGenerator with the karas_imagen parameter when using
    # image augmentation!!
    # --------------------------------------------------------------------------------------------

    num_batches = int(data.shape[0] / batch_size)
    if (data.shape[0] % batch_size != 0):
        num_batches += 1

    train_op, cost, accuracy = model['train_op'], model['loss'], model['accuracy']

    loss, acc = 0.0, 0.0

    eta = None
    cum_time = 0.0

    # get the placeholders from the graph
    try:
        tf_graph = tf.get_default_graph()
        X = tf_graph.get_tensor_by_name("X:0")
        y = tf_graph.get_tensor_by_name("y:0")
    except KeyError as err:
        # could not find the tensors with this name in the graph!!
        print(err, flush=True)
        raise err

    for batch_num in range(num_batches):
        batch_start_time = time.time()
        x_train_batch = data[batch_num * batch_size: min((batch_num + 1) * batch_size, len(data))]
        y_train_batch = labels[batch_num * batch_size: min((batch_num + 1) * batch_size, len(labels))]

        if feed_dict is None:
            feed_dict_batch = {X: x_train_batch, y: y_train_batch}
        else:
            feed_dict_batch = feed_dict
            feed_dict_batch[X] = x_train_batch
            feed_dict_batch[y] = y_train_batch

        if do_training:
            # run a training op & update weights
            sess.run(train_op, feed_dict=feed_dict_batch)

        # calculate batch loss & accuracy
        train_loss_batch, train_acc_batch = sess.run([cost, accuracy], feed_dict=feed_dict_batch)
        # loss += (batch_size * train_loss_batch)
        # acc += (batch_size * train_acc_batch)
        # accummulate batch loss & acc, which we will average before returning from function
        loss += train_loss_batch
        acc += train_acc_batch
        batch_end_time = time.time()

        # Calculate ETA: We calculate average time per step so far & extrapolate
        cum_time += (batch_end_time - batch_start_time)
        time_per_batch = cum_time / (batch_num + 1)
        batches_remaining = num_batches - (batch_num + 1)
        eta = time_per_batch * batches_remaining

        if show_progbar:
            __update_progbar(batch_num=batch_num + 1, num_batches=num_batches, train_cost=train_loss_batch,
                             train_acc=train_acc_batch, phase=phase, final=final, eta=eta)

    # loss /= data.shape[0]
    # acc /= data.shape[0]
    # average out the loss & accuracies, which we accummulated across num_batches
    loss /= num_batches
    acc /= num_batches
    # print('Before returning loss = {}, acc = {}'.format(loss, acc), flush=True)
    return loss, acc

def __batch_run_model(sess, model, train_data, train_labels, feed_dict=None, num_epochs=10, batch_size=32,
                      validation_data=None, do_training=True, show_progbar=True):
    # note validation_data is a tuple (val_data, val_labels)
    history_val = {
        'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []
    }
    history_no_val = {
        'loss': [], 'acc': [],
    }

    val_data, val_labels = None, None

    if validation_data is not None:
        if type(validation_data) is not tuple:
            raise ValueError('Invalid data type: The validation_data parameter must be a tuple or None!')
        else:
            val_data, val_labels = validation_data

    history = (history_no_val if validation_data is None else history_val)
    validating = (False if validation_data is None else True)
    num_batches = int(train_data.shape[0] / batch_size) + 1

    if do_training:
        if validating:
            print('Train on %d samples, validate with %d samples. Train for %d epochs, with %d batches per '
                  'epoch.' % (len(train_data), len(val_data), num_epochs, num_batches))
        else:
            print('Train on %d samples. Train for %d epochs,, with %d batches per epoch.' % (
                len(train_data), num_epochs, num_batches))

    for epoch in range(num_epochs):
        if num_epochs > 1:
            print('Epoch %d/%d:' % (epoch + 1, num_epochs))

        #    def __batch_calc_loss_acc(sess, model, data, labels, feed_dict, batch_size=32,
        #                              do_training=False, show_progbar=True, phase='train', final=False):

        phase_x = ('train' if do_training else 'eval')
        # calculate loss & accuracy on training data set
        train_loss, train_acc = __batch_calc_loss_acc(sess, model, train_data, train_labels,
                                                      feed_dict=feed_dict, batch_size=batch_size,
                                                      do_training=do_training, show_progbar=show_progbar,
                                                      phase=phase_x, final=False)

        if validating:
            # compute costs & accuracies on validation sets if we are also validating
            val_loss, val_acc = __batch_calc_loss_acc(sess, model, val_data, val_labels, feed_dict=feed_dict,
                                                      batch_size=np.minimum(val_data.shape[0], batch_size),
                                                      do_training=False, show_progbar=True, phase='valid',
                                                      final=False)

            __update_progbar(batch_num=num_batches, num_batches=num_batches, train_cost=train_loss,
                             train_acc=train_acc, val_test_cost=val_loss, val_test_acc=val_acc, phase=phase_x,
                             final=True)
            sys.stdout.flush()
            history['loss'].append(train_loss)
            history['acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

        else:
            __update_progbar(batch_num=num_batches, num_batches=num_batches, train_cost=train_loss,
                             train_acc=train_acc, phase=phase_x, final=True)
            sys.stdout.flush()
            history['loss'].append(train_loss)
            history['acc'].append(train_acc)

    return history

def __batch_run_predict(sess, model, data, feed_dict=None, batch_size=32):
    num_batches = int(data.shape[0] / batch_size) + 1
    predictions = []

    out = model['out']

    # get X placeholder from tensorflow graph
    try:
        tf_graph = tf.get_default_graph()
        X = tf_graph.get_tensor_by_name("X:0")
    except KeyError as err:
        # could not find the tensors with this name in the graph!!
        print(err, flush=True)
        raise err

    for batch_num in range(num_batches):
        # grab the Nth batch
        x_batch = data[batch_num * batch_size: min((batch_num + 1) * batch_size, len(data))]

        if feed_dict is None:
            feed_dict_predict = {X: x_batch}
        else:
            # will be fed dropout rates like {kp1:1.0,kp2:1.0,kp3:1.0}
            # we'll add value for X placeholder
            feed_dict_predict = feed_dict
            feed_dict_predict[X] = x_batch

        pred = sess.run(out, feed_dict=feed_dict_predict)
        predictions.extend(pred.tolist())

    return np.array(predictions)

def __gen_calc_loss_acc(sess, model, data_labels_gen, num_steps, feed_dict=None, do_training=False,
                        show_progbar=True, phase='train', final=False):
    # -------------------------------------------------------------------------------------------
    # batch calculates loss & accuracy using loss & accuracy 'attributes' of the model dict
    # batch calculations preclude OOM errors in Tensorflow - does not avoid them, but you could
    # tweak the batch size to almost eliminate them
    # NOTE: data_labels_gen MUST be an instance of keras.preprocessing.image.NumpyArrayIterator
    # --------------------------------------------------------------------------------------------

    from keras.preprocessing.image import NumpyArrayIterator, DirectoryIterator

    assert isinstance(data_labels_gen, NumpyArrayIterator) or isinstance(data_labels_gen, DirectoryIterator)

    train_op, cost, accuracy = model['train_op'], model['loss'], model['accuracy']

    loss, acc = 0.0, 0.0

    eta = None
    cum_time = 0.0

    # get the placeholders from the graph
    try:
        tf_graph = tf.get_default_graph()
        X = tf_graph.get_tensor_by_name("X:0")
        y = tf_graph.get_tensor_by_name("y:0")
    except KeyError as err:
        # could not find the tensors with this name in the graph!!
        print(err, flush=True)
        raise err

    for step in range(num_steps):
        step_start_time = time.time()
        # grab the Nth batch from generator
        x_train_batch, y_train_batch = next(data_labels_gen)

        if feed_dict is None:
            feed_dict_batch = {X: x_train_batch, y: y_train_batch}
        else:
            # NOTE: feed_dict = {dict of keep_probabilities}
            feed_dict_batch = feed_dict
            feed_dict_batch[X] = x_train_batch
            feed_dict_batch[y] = y_train_batch

        if do_training:
            # run a training op & update weights
            sess.run(train_op, feed_dict=feed_dict_batch)

        # calculate batch loss & accuracy
        train_step_loss, train_step_acc = sess.run([cost, accuracy], feed_dict=feed_dict_batch)
        loss += train_step_loss
        acc += train_step_acc
        step_end_time = time.time()

        # Calculate ETA: We calculate average time per step so far & extrapolate
        cum_time += (step_end_time - step_start_time)
        time_per_step = cum_time / (step + 1)
        steps_remaining = num_steps - (step + 1)
        eta = time_per_step * steps_remaining

        if show_progbar:
            __update_progbar(batch_num=step + 1, num_batches=num_steps, train_cost=train_step_loss,
                             train_acc=train_step_acc, phase=phase, final=final, eta=eta)

    loss /= num_steps
    acc /= num_steps

    return loss, acc

def __gen_run_model(sess, model, data_labels_gen, steps_per_epoch, num_epochs=15, feed_dict=None,
                    validation_data=None, validation_steps=None, do_training=True, show_progbar=True):
    from keras.preprocessing.image import NumpyArrayIterator, DirectoryIterator
    assert isinstance(data_labels_gen, NumpyArrayIterator) or isinstance(data_labels_gen, DirectoryIterator)
    if validation_data is not None:
        assert ((type(validation_data) is tuple) or isinstance(data_labels_gen,
                                                               NumpyArrayIterator) or isinstance(
            data_labels_gen, DirectoryIterator))

    # note validation_data is either a generator or tuple (val_data, val_labels)
    history_val = {
        'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []
    }
    history_no_val = {
        'loss': [], 'acc': [],
    }

    val_from_gen = True
    val_data, val_labels = None, None
    steps_per_epoch = int(steps_per_epoch)
    if validation_steps is not None:
        validation_steps = int(validation_steps)

    if validation_data is not None:
        if type(validation_data) is tuple:
            val_data, val_labels = validation_data
            val_from_gen = False
        else:  # (isinstance(validation_data, NumpyArrayIterator | DirectoryIterator)):
            # validation_steps is required!
            assert ((validation_steps is not None) and (validation_steps > 0))

    history = (history_no_val if validation_data is None else history_val)
    validating = (False if validation_data is None else True)

    for epoch in range(num_epochs):
        if num_epochs > 1:
            print('Epoch %d/%d:' % (epoch + 1, num_epochs))

        phase_x = ('train' if do_training else 'eval')

        # calculate loss & accuracy on training data set
        train_loss, train_acc = __gen_calc_loss_acc(sess, model, data_labels_gen, num_steps=steps_per_epoch,
                                                    feed_dict=feed_dict, do_training=do_training,
                                                    show_progbar=show_progbar, phase=phase_x, final=False)

        if validating:
            if val_from_gen:
                # if validation_data is an instance of keras.preprocessing.image.NumpyArrayIterator
                val_loss, val_acc = __gen_calc_loss_acc(sess, model, validation_data, validation_steps,
                                                        feed_dict=feed_dict, do_training=False,
                                                        show_progbar=True, phase='valid')
            else:
                # if validation_data is passed in as a tuple. We'll assume a batch size of 32 or
                # size of the validation data set, whichever is lesser
                val_loss, val_acc = __batch_calc_loss_acc(sess, model, val_data, val_labels,
                                                          np.minimum(val_data.shape[0], 32),
                                                          feed_dict=feed_dict, do_training=False,
                                                          show_progbar=True, phase='valid', final=False)

            __update_progbar(batch_num=steps_per_epoch, num_batches=steps_per_epoch, train_cost=train_loss,
                             train_acc=train_acc, val_test_cost=val_loss, val_test_acc=val_acc, phase=phase_x,
                             final=True)
            sys.stdout.flush()
            history['loss'].append(train_loss)
            history['acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
        else:
            __update_progbar(batch_num=steps_per_epoch, num_batches=steps_per_epoch, train_cost=train_loss,
                             train_acc=train_acc, phase=phase_x, final=True)
            sys.stdout.flush()
            history['loss'].append(train_loss)
            history['acc'].append(train_acc)
    return history

# -----------------------------------------------------------------------------------------------
# Tensorflow model execution API
# -----------------------------------------------------------------------------------------------
def train_model(sess, model, data, labels, feed_dict=None, num_epochs=10, batch_size=32,
                validation_split=None, validation_data=None):
    """
    trains the model on data & labels and cross-validates on validation_data, if provided. The
    model is run through batch gradient descent, using a batch size = batch_size (=32 by default)
    for num_epochs (=10 by default) epochs.
    @params:
        - sess: instance of Tensorflow Session object 
        - model: the Tensorflow model (a dict with tensor operations). The model is expected to have named
            attributes 'train_op', 'loss', 'accuracy', 'out' etc. defined
        - data & labels: training data & labels (i.e. X_train, y_train in scikit-learn parlance)
        - feed_dict - any placeholder values that need to be passed to model during evaluation
          This feed dict is usually used only when we have to pass dropout values or other such
          placeholders. Do not pass X & y.
        - num_epochs: number of epochs through which training is run (optional, default = 10)
        - batch_size: size of a batch used during batch training (optional, default = 32)
          As a general practice, use a smaller batch, like 10-32. Though training time increases, you 
          land up using lesser memory during training.
        - validation_split (optional floating point value): fraction  of the training data that must 
          be set aside as the validation data. E.g. if validation_split=0.1, then 10% of the
          data & labels (training set) is randomly selected & set aside as validation data & balance 90%
          is used as the training set.
        - validation_data: a tuple of (val_data, val_labels) that define the validation set against
          which the model will cross-validate (optional). You can use this INSTEAD of using
          validation_split.
          NOTE:
            - use either validation_data or validation_split. If BOTH are used then validation_split
              is given preference & validation_data is ignored.
            - it is always a good idea to shuffle the training/validation data before calling
              this function, unless data must be in some sequence!
    """

    if validation_split is not None:
        num_val_recs = int(validation_split * data.shape[0])
        # set-aside num_val_recs as validation set from data, rest is data
        val_data = data[:num_val_recs]
        val_labels = labels[:num_val_recs]
        data2 = data[num_val_recs:]
        labels2 = labels[num_val_recs:]

        history = __batch_run_model(sess, model, data2, labels2, feed_dict=feed_dict, num_epochs=num_epochs,
                                    batch_size=batch_size, validation_data=(val_data, val_labels),
                                    do_training=True, show_progbar=True)
    else:
        history = __batch_run_model(sess, model, data, labels, feed_dict=feed_dict, num_epochs=num_epochs,
                                    batch_size=batch_size, validation_data=validation_data, do_training=True,
                                    show_progbar=True)
    return history

def fit_model(sess, model, data, labels, feed_dict=None, num_epochs=10, batch_size=32, validation_split=None,
              validation_data=None):
    """ an alternate function for those familiar with Keras - just calls train_model()
        with all parameters as-is """
    return train_model(sess, model, data, labels, feed_dict=feed_dict, num_epochs=num_epochs,
                       batch_size=batch_size, validation_split=validation_split,
                       validation_data=validation_data)

def evaluate_model(sess, model, data, labels, feed_dict=None, batch_size=32):
    """ used to evaluate model's performance against data (i.e. calculate loss & accuracy)
        Will run the model for 1 epoch across provided data & labels set & evaluate loss & accuracy metrics
        @params:
            - sess: instance of Tensorflow Session() object
            - model: model to evaluate
            - data, labels: evaluation data & labels
            - feed_dict (optional, default=None): feed dictionary to pass in any placeholders to model
            - batch_size (optional, default=32): batch size to use for training
               (if len(data) < batch_size, the len(data) is used as batch size)
        @returns:
            - tuple with calculated avarage loss & accuracy values (avg_loss, avg_acc)
    """
    history = __batch_run_model(sess, model, data, labels, feed_dict=feed_dict, num_epochs=1,
                                batch_size=batch_size, do_training=False, show_progbar=True,
                                validation_data=None)
    avg_loss, avg_acc = np.mean(history['loss']), np.mean(history['acc'])
    return (avg_loss, avg_acc)

def predict(sess, model, data, feed_dict=None):
    """ return predictions as probabilities """
    predictions = __batch_run_predict(sess, model, data, feed_dict=feed_dict,
                                      batch_size=np.minimum(data.shape[0], 32))
    return predictions  # np.argmax(predictions, axis=1)

# def predict_probs(sess, model, data, feed_dict=None):
#    predictions = __batch_run_predict(sess, model, data, feed_dict=feed_dict,
#                                      batch_size=np.minimum(data.shape[0], 32))
#    # convert log values to probabilities - take inverse logs
#    probabilities = sess.run(tf.nn.softmax(predictions))
#    return np.argmax(predictions, axis=1), probabilities

# ------------------------------------------------------------------------------------------------
# functions using Keras' ImageDataGenerator class
# ------------------------------------------------------------------------------------------------
def fit_generator(sess, model, data_labels_gen, steps_per_epoch, num_epochs=15, feed_dict=None,
                  validation_data=None, validation_steps=None):
    # ---------------------------------------------------------------------------------------------
    # validation_split defines the %age of 'data' that must be set aside as validation set/
    # if validation_split is NOT None then validation_data is ignored (i.e. validation_split is
    # given preference!)
    # Instead of validation_split, you can also pass a tuple (val_data, val_labels) in the
    # validation_data
    # parameter. You'll then need to 'create' the validation data before calling this function.
    # NOTE: it is always good to shuffle the data/labels before calling this function.
    # Alternatively,
    # you can set shuffle=True to ask this function to shuffle the data/labels
    # ---------------------------------------------------------------------------------------------

    history = __gen_run_model(sess, model, data_labels_gen, steps_per_epoch=steps_per_epoch,
                              num_epochs=num_epochs, feed_dict=feed_dict, validation_data=validation_data,
                              validation_steps=validation_steps, do_training=True, show_progbar=True)

    return history

def evaluate_generator(sess, model, data_labels_gen, steps_per_epoch, feed_dict=None):
    # ---------------------------------------------------------------------------------------------
    # validation_split defines the %age of 'data' that must be set aside as validation set/
    # if validation_split is NOT None then validation_data is ignored (i.e. validation_split is
    # given preference!)
    # Instead of validation_split, you can also pass a tuple (val_data, val_labels) in the
    # validation_data
    # parameter. You'll then need to 'create' the validation data before calling this function.
    # NOTE: it is always good to shuffle the data/labels before calling this function.
    # Alternatively,
    # you can set shuffle=True to ask this function to shuffle the data/labels
    # ---------------------------------------------------------------------------------------------

    history = __gen_run_model(sess, model, data_labels_gen, steps_per_epoch=steps_per_epoch, num_epochs=1,
                              feed_dict=feed_dict, validation_data=None, validation_steps=None,
                              do_training=False, show_progbar=True)
    avg_loss, avg_acc = np.mean(history['loss']), np.mean(history['acc'])
    return (avg_loss, avg_acc)

def predict_generator(sess, model, data_labels_gen, steps_per_epoch, feed_dict=None):
    from keras.preprocessing.image import NumpyArrayIterator

    assert isinstance(data_labels_gen, NumpyArrayIterator)
    predictions = []
    actuals = []

    out = model['out']

    # get the placeholders from the graph
    try:
        tf_graph = tf.get_default_graph()
        X = tf_graph.get_tensor_by_name("X:0")
    except KeyError as err:
        # could not find the tensors with this name in the graph!!
        print(err, flush=True)
        raise err

    for step in steps_per_epoch:
        # grab a batch from the data generator
        x_batch, y_true_batch = next(data_labels_gen)

        if feed_dict is None:
            feed_dict_predict = {X: x_batch}
        else:
            # will be fed dropout rates like {kp1:1.0,kp2:1.0,kp3:1.0}
            # we'll add value for X placeholder
            feed_dict_predict = feed_dict
            feed_dict_predict[X] = x_batch

        y_pred_batch = sess.run(out, feed_dict=feed_dict_predict)

        predictions.extend(y_pred_batch.tolist())
        actuals.extend(y_true_batch.tolist())
        __progbar_msg(step, steps_per_epoch, 'Predicting', '')
    __progbar_msg(steps_per_epoch, steps_per_epoch, 'Predicting', '', final=True)

    # NOTE: each member of the predictions list will be an array of (,num_classes) dimensions
    # whereas the actuals will have a single value
    return np.array(predictions, actuals)

# ------------------------------------------------------------------------------------------------
# loading & saving Tensorflow graphs from persistent state
# ------------------------------------------------------------------------------------------------
def save_tf_model(sess, base_file_name='network', model_path='./model_states'):
    # save the default graph (@see: https://stackoverflow.com/questions/51322381/tensorflow-model-saving
    # -and-loading
    try:
        train_writer = tf.summary.FileWriter(model_path)
        train_writer.add_graph(tf.get_default_graph())
        # save the weights of the model
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(model_path, base_file_name))
    except IOError as err:
        print('Error saving Tensorflow graph -> {}'.format(err), flush=True)
        raise err

def load_tf_model(sess, base_file_name='network', model_path='./model_states'):
    # load the model from persistent state
    # @see: https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
    # @see: https://stackoverflow.com/questions/42685994/how-to-get-a-tensorflow-op-by-name
    try:
        meta_path = os.path.join(model_path, base_file_name + '.meta')
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        tf_graph = tf.get_default_graph()
        return tf_graph
    except IOError as err:
        print('Error loading Tensorflow graph -> {}'.format(err), flush=True)
        raise err
