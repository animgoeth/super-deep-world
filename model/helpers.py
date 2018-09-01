import tensorflow as tf
import numpy as np


def masked_conv2d(inputs,
                  input_channels_count,
                  filters_count,
                  kernel_size,
                  mask_type,
                  channels_masked,
                  padding='valid',
                  activation=None,
                  kernel_initializer=None,
                  name=None):
    with tf.variable_scope(name):
        weights_shape = (kernel_size, kernel_size, input_channels_count, filters_count)
        weights = tf.get_variable("weights", weights_shape, tf.float32, kernel_initializer)

        if mask_type is not None:
            center_h = kernel_size // 2
            center_w = kernel_size // 2

            mask = np.ones(weights_shape, dtype=np.float32)

            mask[center_h, center_w + 1:, :, :] = 0.
            mask[center_h + 1:, :, :, :] = 0.

            if mask_type == 'A':
                mask[center_h, center_w, :channels_masked, :] = 0.

            weights = weights * tf.constant(mask, dtype=tf.float32)

        outputs = tf.nn.conv2d(inputs, weights, [1, 1, 1, 1], padding=padding, name='outputs')
        biases = tf.get_variable("biases", [filters_count, ], tf.float32, tf.zeros_initializer())
        outputs = tf.nn.bias_add(outputs, biases, name='outputs_plus_b')

        if activation:
            outputs = activation(outputs, name='outputs_with_fn')

        return outputs


class ConvRNNCell(object):
    def __call__(self, inputs, state, scope=None):
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size):
        shape = self.shape
        num_features = self.num_features
        zeros = tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, shape[0], shape[1], num_features]),
                                              tf.zeros([batch_size, shape[0], shape[1], num_features]))
        return zeros


class BasicConvLSTMCell(ConvRNNCell):
    def __init__(self, shape, filter_size, num_features, forget_bias=1.0,
                 state_is_tuple=True, activation=tf.nn.tanh, initializer=None):
        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._initializer = initializer

    @property
    def state_size(self):
        return (tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope):
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
            concat = conv_linear([inputs, h], self.filter_size, self.num_features * 4,
                                 True, scope=scope, initializer=self._initializer)

            input_gate, new_input_gate, forget_gate, output_gate = tf.split(axis=3, num_or_size_splits=4, value=concat)

            new_c = (c * tf.nn.sigmoid(forget_gate + self._forget_bias) + tf.nn.sigmoid(input_gate) * self._activation(new_input_gate))
            new_h = self._activation(new_c) * tf.nn.sigmoid(output_gate)

            if self._state_is_tuple:
                new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(axis=3, values=[new_c, new_h])

            return new_h, new_state


def conv_linear(inputs, filter_size, num_features, bias, bias_start=0.0, scope=None, initializer=None):
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in inputs]

    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in inputs][0]

    with tf.variable_scope(scope):
        matrix = tf.get_variable(
            name="Matrix",
            shape=[filter_size[0], filter_size[1], total_arg_size_depth, num_features],
            dtype=dtype,
            initializer=initializer)

        res = tf.nn.conv2d(
            input=inputs[0] if len(inputs) == 1 else tf.concat(axis=3, values=inputs),
            filter=matrix,
            strides=[1, 1, 1, 1],
            padding='SAME')

        if not bias:
            return res

        bias_term = tf.get_variable(
            name="Bias",
            shape=[num_features],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))

    return res + bias_term
