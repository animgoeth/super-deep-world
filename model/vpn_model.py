import tensorflow as tf
from model.helpers import masked_conv2d, BasicConvLSTMCell
from infra.logger import Logger


class VideoPixelNetworkModel(object):
    def __init__(self, config):
        self.config = config

        with tf.name_scope('inputs'):
            self.sequences = tf.placeholder(dtype=tf.float32,
                                            shape=[None, config.truncated_steps + 1] + config.input_shape,
                                            name='sequences')
            self.inference_prev_frame = tf.placeholder(dtype=tf.float32,
                                                       shape=[None] + config.input_shape,
                                                       name='inference_prev_frame')
            self.inference_encoder_state = tf.placeholder(dtype=tf.float32,
                                                          shape=[None,
                                                                 config.input_shape[0],
                                                                 config.input_shape[1],
                                                                 config.conv_lstm_filters],
                                                          name='inference_encoder_state')
            self.inference_current_frame = tf.placeholder(dtype=tf.float32,
                                                          shape=[None] + config.input_shape,
                                                          name='inference_current_frame')
            self.initial_lstm_state = tf.placeholder(dtype=tf.float32,
                                                     shape=[2,
                                                            None,
                                                            config.input_shape[0],
                                                            config.input_shape[1],
                                                            config.conv_lstm_filters],
                                                     name='initial_lstm_state')
        self.build_model()

    def multiplicative_unit_without_mask(self, h, dilation_rate, scope):
        with tf.variable_scope('multiplicative_unit_without_mask_' + scope):
            g1 = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                3,
                dilation_rate=dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g1'
            )

            g2 = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                3,
                dilation_rate=dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g2'
            )

            g3 = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                3,
                dilation_rate=dilation_rate,
                padding='same',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g3'
            )

            u = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                3,
                dilation_rate=dilation_rate,
                padding='same',
                activation=tf.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='u'
            )

            g2_h = tf.multiply(g2, h)
            g3_u = tf.multiply(g3, u)

            mu = tf.multiply(g1, tf.tanh(g2_h + g3_u))

            return mu

    def multiplicative_unit_with_mask(self, h, mask_type, masked_channels, in_channels, out_channels, scope):
        with tf.variable_scope('multiplicative_unit_with_mask_' + scope):
            g1 = masked_conv2d(
                h,
                in_channels,
                out_channels,
                3,
                mask_type,
                masked_channels,
                padding='SAME',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g1')

            g2 = masked_conv2d(
                h,
                in_channels,
                out_channels,
                3,
                mask_type,
                masked_channels,
                padding='SAME',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g2')

            g3 = masked_conv2d(
                h,
                in_channels,
                out_channels,
                3,
                mask_type,
                masked_channels,
                padding='SAME',
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='g3')

            u = masked_conv2d(
                h,
                in_channels,
                out_channels,
                3,
                mask_type,
                masked_channels,
                padding='SAME',
                activation=tf.tanh,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='u')

            g2_h = tf.multiply(g2, h)
            g3_u = tf.multiply(g3, u)

            if mask_type == 'B':
                g2_h = tf.multiply(g2, h)
            else:
                g2_h = g2
            g3_u = tf.multiply(g3, u)

            mu = tf.multiply(g1, tf.tanh(g2_h + g3_u))

            return mu

    def residual_multiplicative_block_without_mask(self, h, dilation_rate, scope):
        with tf.variable_scope('residual_multiplicative_block_without_mask_' + scope):
            h1 = tf.layers.conv2d(
                h,
                self.config.rmb_c,
                1,
                padding='same',
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='h1'
            )
            h2 = self.multiplicative_unit_without_mask(h1, dilation_rate, '1')
            h3 = self.multiplicative_unit_without_mask(h2, dilation_rate, '2')
            h4 = tf.layers.conv2d(
                h3,
                2 * self.config.rmb_c,
                1,
                padding='same',
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='h4'
            )

            rmb = tf.add(h, h4)

            return rmb

    def residual_multiplicative_block_with_mask(self, h, scope, first_block=False, last_block=False):
        with tf.variable_scope('residual_multiplicative_block_with_mask_' + scope):

            if first_block:
                x, h = h
                h1 = tf.layers.conv2d(
                    h,
                    self.config.rmb_c - self.config.input_shape[2],
                    1,
                    padding='same',
                    activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name='h1'
                )
                h1 = tf.concat([x, h1], axis=3)
            else:
                h1 = tf.layers.conv2d(
                    h,
                    self.config.rmb_c,
                    1,
                    padding='same',
                    activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name='h1'
                )

            if first_block:
                h2 = self.multiplicative_unit_with_mask(
                    h1, 'A',
                    self.config.input_shape[2],
                    self.config.rmb_c,
                    self.config.rmb_c,
                    '1')
            else:
                h2 = self.multiplicative_unit_with_mask(
                    h1, 'B',
                    self.config.rmb_c,
                    self.config.rmb_c,
                    self.config.rmb_c,
                    '1')

            h3 = self.multiplicative_unit_with_mask(
                h2, 'B',
                self.config.rmb_c,
                self.config.rmb_c,
                self.config.rmb_c,
                '2')

            if last_block:
                rmb = tf.layers.conv2d(
                    h3,
                    256,
                    1,
                    padding='same',
                    activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name='h4'
                )
            else:
                h4 = tf.layers.conv2d(
                    h3,
                    2 * self.config.rmb_c,
                    1,
                    padding='same',
                    activation=None,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name='h4'
                )
                rmb = tf.add(h, h4)

            return rmb

    def resolution_preserving_cnn_encoders(self, x):
        with tf.variable_scope('resolution_preserving_cnn_encoders'):
            x = tf.layers.conv2d(
                x,
                2 * self.config.rmb_c,
                1,
                padding='same',
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='input_conv'
            )

            for i in range(self.config.encoder_rmb_num):
                x = self.residual_multiplicative_block_without_mask(x, 1, str(i))

            return x

    def pixel_cnn_decoders(self, h, x):
        with tf.variable_scope('pixel_cnn_decoders'):
            h = self.residual_multiplicative_block_without_mask(h, 1, '00')

            h = (x, h)
            h = self.residual_multiplicative_block_with_mask(h, str(1), first_block=True)

            for i in range(2, self.config.decoder_rmb_num - 1):
                h = self.residual_multiplicative_block_with_mask(h, str(i))

            h = self.residual_multiplicative_block_with_mask(h, str(self.config.decoder_rmb_num - 1), last_block=True)

            return h

    def conv_lstm(self, x, h):
        with tf.variable_scope('Conv_LSTM') as scope:
            convlstm_cell = BasicConvLSTMCell(
                [self.config.input_shape[0], self.config.input_shape[1]],
                [3, 3],
                self.config.conv_lstm_filters,
                initializer=tf.contrib.layers.xavier_initializer())

            output, state = convlstm_cell(x, h, scope)

            return output, state

    def encoder_template(self, x, lstm_h):
        encoder_h = self.resolution_preserving_cnn_encoders(x)
        encoder_h, lstm_h = self.conv_lstm(encoder_h, lstm_h)
        return encoder_h, lstm_h

    def decoder_template(self, encoder_h, x_):
        output = self.pixel_cnn_decoders(encoder_h, x_)
        return output

    def build_model(self):
        lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state[0], self.initial_lstm_state[1])
        encoder_network_template = tf.make_template('vpn_encoder', self.encoder_template)
        decoder_network_template = tf.make_template('vpn_decoder', self.decoder_template)

        with tf.name_scope('training_graph'):
            net_unwrap = []
            for i in range(self.config.truncated_steps):
                encoder_state, lstm_state = encoder_network_template(self.sequences[:, i], lstm_state)
                step_out = decoder_network_template(encoder_state, self.sequences[:, i + 1])
                net_unwrap.append(step_out)

            self.final_lstm_state = lstm_state

        with tf.name_scope('wrap_out'):
            net_unwrap = tf.stack(net_unwrap)
            self.output = tf.transpose(net_unwrap, [1, 0, 2, 3, 4])

            for i in range(self.config.truncated_steps):
                Logger.summarize_images(tf.expand_dims(tf.cast(tf.arg_max(self.output[:, i], 3), tf.float32), 3),
                                        'frame_{0}'.format(i),
                                        'vpn',
                                        1)

        with tf.name_scope('loss'):
            labels = tf.one_hot(indices=tf.cast(tf.squeeze(self.sequences[:, 1:]), tf.int32),
                                depth=256,
                                axis=-1,
                                dtype=tf.float32)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=labels))
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope('inference_graph'):
            lstm_state = tf.contrib.rnn.LSTMStateTuple(self.initial_lstm_state[0], self.initial_lstm_state[1])
            self.encoder_state, lstm_state = encoder_network_template(self.inference_prev_frame, lstm_state)
            self.inference_lstm_state = lstm_state
            self.inference_output = decoder_network_template(self.inference_encoder_state, self.inference_current_frame)

        with tf.name_scope('test_frames'):
            self.test_summaries = []
            for i in range(self.config.truncated_steps + 1):
                Logger.summarize_images(tf.expand_dims(tf.cast(tf.arg_max(self.inference_output, 3), tf.float32), 3),
                                        'test_frame_{0}'.format(i),
                                        'vpn_test_{0}'.format(i),
                                        1)
                self.test_summaries.append(tf.summary.merge_all('vpn_test_{0}'.format(i)))

        self.summaries = tf.summary.merge_all('vpn')
