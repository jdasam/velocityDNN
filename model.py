from __future__ import division

import tensorflow as tf
import numpy as np
import time
from six.moves import xrange
import io
import matplotlib.pyplot as plt
import evaluate
import utils
from StringIO import StringIO


def lrelu(x, alpha=0.2, name='lrelu'):
    with tf.variable_scope(name):
        return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

class Model:
    # TODO: move summaries to callback class.
    def __init__(self, sess, config, pipeline):
        self.sess = sess
        self.config = config
        self.pipeline = pipeline
        # with tf.variable_scope('place_holders') as scope:
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.keep_prob = tf.placeholder(tf.float32, [len(config.dropout)], name='dropout_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='lr')
        self.dropout = config.dropout
        self.n_drop = 0
        self.epoch = 0

        self.input = pipeline.input
        self.label = pipeline.label
        self.current_learning_rate = config.learning_rate
        self.handle = pipeline.handle
        self.train_handle = sess.run(pipeline.train_iterator.string_handle())
        self.valid_handle = sess.run(pipeline.valid_iterator.string_handle())
        self.test_handle = sess.run(pipeline.test_iterator.string_handle())

        # TODO: seperate this part
        with tf.variable_scope('input') as scope:
            input = tf.identity(self.input, name='input')

        if config.type == 'cnn_type_1':
            input = tf.expand_dims(input, -1)
            # conv1
            with tf.variable_scope('conv1') as scope:
                kernel = tf.get_variable('weight', shape=[3, 3, 1, 8],
                                         initializer=tf.contrib.layers.xavier_initializer())
                input = tf.pad(input, [[0, 0], [0, 0], [1, 1], [0, 0]], 'CONSTANT')
                conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='VALID')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # conv2
            with tf.variable_scope('conv2') as scope:
                kernel = tf.get_variable('weight', shape=[3, 3, 8, 8],
                                         initializer=tf.contrib.layers.xavier_initializer())
                activation = tf.pad(activation, [[0, 0], [0, 0], [1, 1], [0, 0]], 'CONSTANT')
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='VALID')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # conv3
            with tf.variable_scope('conv3') as scope:
                kernel = tf.get_variable('weight', shape=[1, 20, 8, 32],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='SAME')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # output
            with tf.variable_scope('output') as scope:
                kernel = tf.get_variable('weight', shape=[1, self.config.feature_dim, 32, self.config.output_dim],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='VALID')
                pred = tf.reshape(conv, [-1, self.config.seg_length - 4, self.config.output_dim], name='pred')

        elif config.type == 'cnn_type_2':
            input = tf.expand_dims(input, -1)
            # conv1
            with tf.variable_scope('conv1') as scope:
                kernel = tf.get_variable('weight', shape=[3, 3, 1, 16],
                                         initializer=tf.contrib.layers.xavier_initializer())
                input = tf.pad(input, [[0, 0], [0, 0], [1, 1], [0, 0]], 'CONSTANT')
                conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='VALID')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # conv2
            with tf.variable_scope('conv2') as scope:
                kernel = tf.get_variable('weight', shape=[3, 3, 16, 16],
                                         initializer=tf.contrib.layers.xavier_initializer())
                activation = tf.pad(activation, [[0, 0], [0, 0], [1, 1], [0, 0]], 'CONSTANT')
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='VALID')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # conv2_2
            with tf.variable_scope('conv2_2') as scope:
                kernel = tf.get_variable('weight', shape=[1, 3, 16, 16],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='SAME')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))


            # conv3
            with tf.variable_scope('conv3') as scope:
                kernel = tf.get_variable('weight', shape=[1, 20, 16, 32],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='SAME')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))


            with tf.variable_scope('output') as scope:
                kernel = tf.get_variable('weight', shape=[1, self.config.feature_dim, 32, self.config.output_dim],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='VALID')
                pred = tf.reshape(conv, [-1, self.config.seg_length - 4, self.config.output_dim], name='pred')

        elif config.type == 'cnn_type_3':
            input = tf.expand_dims(input, -1)
            # conv1
            with tf.variable_scope('conv1') as scope:
                kernel = tf.get_variable('weight', shape=[3, 3, 1, 32],
                                         initializer=tf.contrib.layers.xavier_initializer())
                input = tf.pad(input, [[0, 0], [0, 0], [1, 1], [0, 0]], 'CONSTANT')
                conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='VALID')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # conv2
            with tf.variable_scope('conv2') as scope:
                kernel = tf.get_variable('weight', shape=[3, 3, 32, 32],
                                         initializer=tf.contrib.layers.xavier_initializer())
                activation = tf.pad(activation, [[0, 0], [0, 0], [1, 1], [0, 0]], 'CONSTANT')
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='VALID')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # conv2_2
            with tf.variable_scope('conv2_2') as scope:
                kernel = tf.get_variable('weight', shape=[1, 3, 32, 32],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='SAME')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))


            # conv3
            with tf.variable_scope('conv3') as scope:
                kernel = tf.get_variable('weight', shape=[1, 3, 32, 32],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='SAME')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # conv4
            with tf.variable_scope('fc1') as scope:
                kernel = tf.get_variable('weight', shape=[1, self.config.feature_dim, 32, 256],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='VALID')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # output
            with tf.variable_scope('output') as scope:
                kernel = tf.get_variable('weight', shape=[1, 1, 256, self.config.output_dim],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='VALID')
                pred = tf.reshape(conv, [-1, self.config.seg_length-4, self.config.output_dim], name='pred')

        elif config.type == 'cnn_type_4':
            def cnn(input):
                # conv1
                with tf.variable_scope('conv1') as scope:
                    # input = tf.reshape(input, shape=[-1, 5, 39, 1])
                    kernel = tf.get_variable('weight', shape=[3, 3, 1, 32],
                                             initializer=tf.contrib.layers.xavier_initializer())
                    input = tf.pad(input, [[0, 0], [0, 0], [1, 1], [0, 0]], 'CONSTANT')
                    conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='VALID')
                    batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                              is_training=self.is_train)
                    activation = tf.nn.relu(batch_norm, name='activation')
                    tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

                # conv2
                with tf.variable_scope('conv2') as scope:
                    kernel = tf.get_variable('weight', shape=[3, 3, 32, 32],
                                             initializer=tf.contrib.layers.xavier_initializer())
                    activation = tf.pad(activation, [[0, 0], [0, 0], [1, 1], [0, 0]], 'CONSTANT')
                    conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='VALID')
                    batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                              is_training=self.is_train)
                    activation = tf.nn.relu(batch_norm, name='activation')
                    tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

                # conv2_2
                with tf.variable_scope('conv2_2') as scope:
                    kernel = tf.get_variable('weight', shape=[1, 3, 32, 32],
                                             initializer=tf.contrib.layers.xavier_initializer())
                    conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='SAME')
                    batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                              is_training=self.is_train)
                    activation = tf.nn.relu(batch_norm, name='activation')
                    tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))


                # conv3
                with tf.variable_scope('conv3') as scope:
                    kernel = tf.get_variable('weight', shape=[1, 3, 32, 32],
                                             initializer=tf.contrib.layers.xavier_initializer())
                    conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='SAME')
                    batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                              is_training=self.is_train)
                    activation = tf.nn.relu(batch_norm, name='activation')
                    tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

                # fc1
                with tf.variable_scope('fc1') as scope:
                    dim = 32 * 39
                    reshape = tf.reshape(activation, [-1, dim])
                    weights = tf.get_variable('weights', shape=[dim, 256],
                                              initializer=tf.contrib.layers.xavier_initializer())
                    biases = tf.get_variable('biases', [256], initializer=tf.constant_initializer(0.0))
                    activation = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='activation')
                    tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(weights), self.config.regularize))

                with tf.variable_scope('output') as scope:
                    weights = tf.get_variable('weights', shape=[256, 44],
                                              initializer=tf.contrib.layers.xavier_initializer())
                    biases = tf.get_variable('biases', [44], initializer=tf.constant_initializer(0.0))
                    output = tf.add(tf.matmul(activation, weights), biases, name='output')
                return output

            self.input = tf.expand_dims(self.input, -1)
            inputs = []
            labels = []
            with tf.variable_scope('cnn') as scope:
                for n in xrange(100-4):
                    input_frag = self.input[:, n: n + 5, :, :]
                    label_frag = self.label[:, n, :]
                    inputs.append(input_frag)
                    labels.append(label_frag)
            inputs = tf.concat(inputs, axis=0)
            self.label = tf.concat(labels, axis=0)
            pred = cnn(inputs)

        elif config.type == 'cnn_type_5':
            self.input = tf.expand_dims(self.input, -1)
            # conv1
            with tf.variable_scope('conv1') as scope:
                kernel = tf.get_variable('weight', shape=[3, 3, 1, 16],
                                         initializer=tf.contrib.layers.xavier_initializer())
                input = tf.pad(self.input, [[0, 0], [0, 0], [1, 1], [0, 0]], 'CONSTANT')
                conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='VALID')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                             is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # conv2
            with tf.variable_scope('conv2') as scope:
                kernel = tf.get_variable('weight', shape=[3, 3, 16, 16],
                                         initializer=tf.contrib.layers.xavier_initializer())
                activation = tf.pad(activation, [[0, 0], [0, 0], [1, 1], [0, 0]], 'CONSTANT')
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='VALID')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # conv2_2
            with tf.variable_scope('conv2_2') as scope:
                kernel = tf.get_variable('weight', shape=[1, 3, 16, 16],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='SAME')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # conv3
            with tf.variable_scope('conv3') as scope:
                kernel = tf.get_variable('weight', shape=[1, 20, 16, 32],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='SAME')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # conv4
            with tf.variable_scope('conv4') as scope:
                kernel = tf.get_variable('weight', shape=[1, self.config.feature_dim, 32, self.config.output_dim],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='VALID')
                pred = tf.reshape(conv, [-1, 96, self.config.output_dim])

        elif config.type == 'crnn_type_1':
            input = tf.expand_dims(input, -1)
            # conv1
            with tf.variable_scope('conv1') as scope:
                kernel = tf.get_variable('weight', shape=[3, 3, 1, 8],
                                         initializer=tf.contrib.layers.xavier_initializer())
                input = tf.pad(input, [[0, 0], [0, 0], [1, 1], [0, 0]], 'CONSTANT')
                conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='VALID')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # conv2
            with tf.variable_scope('conv2') as scope:
                kernel = tf.get_variable('weight', shape=[3, 3, 8, 8],
                                         initializer=tf.contrib.layers.xavier_initializer())
                activation = tf.pad(activation, [[0, 0], [0, 0], [1, 1], [0, 0]], 'CONSTANT')
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='VALID')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            # conv3
            with tf.variable_scope('conv3') as scope:
                kernel = tf.get_variable('weight', shape=[1, 20, 8, 32],
                                         initializer=tf.contrib.layers.xavier_initializer())
                conv = tf.nn.conv2d(activation, kernel, [1, 1, 1, 1], padding='SAME')
                batch_norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                          is_training=self.is_train)
                activation = tf.nn.relu(batch_norm, name='activation')
                tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), self.config.regularize))

            with tf.variable_scope('rnn'):
                flatten = tf.reshape(activation, [-1, self.config.seg_length-4, self.config.feature_dim * 32])
                fw = []
                bw = []
                for n in xrange(2):
                    with tf.variable_scope('layer_%d' % n):
                        fw_cell = tf.contrib.rnn.BasicLSTMCell(50, forget_bias=1.0)
                        bw_cell = tf.contrib.rnn.BasicLSTMCell(50, forget_bias=1.0)
                        fw.append(fw_cell)
                        bw.append(bw_cell)

                outputs, state_fw, state_bw = \
                    tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                        fw, bw, flatten, dtype='float32')
                outputs = tf.unstack(outputs, config.seg_length-4, 1)

            with tf.variable_scope('output') as scope:
                weights = tf.get_variable('weights', shape=[100, config.output_dim],
                                          initializer=tf.contrib.layers.xavier_initializer())
                biases = tf.get_variable('biases', [config.output_dim], initializer=tf.constant_initializer(0.0))
                output = [tf.matmul(output, weights) + biases for output in outputs]
                # output = tf.concat(axis=0, values=output)
                pred = tf.transpose(output, [1, 0, 2], name='output')

        sigmoid_pred = tf.sigmoid(pred, name='prediction')

        with tf.name_scope('losses'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=self.label,
                                                                    name='cross_entropy_raw')
            if config.onset_weight:
                cross_entropy = tf.losses.compute_weighted_loss(cross_entropy, pipeline.mask)
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
            tf.add_to_collection('losses', cross_entropy_mean)
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.name_scope('train'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if config.optimizer == 'sgd':
                    optimize = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9,
                                                          use_nesterov=True) \
                        .minimize(total_loss, global_step=global_step)
                elif config.optimizer == 'adam':
                    optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss,
                                                                                                 global_step=global_step)
                else:
                    raise Exception('Undefined optimizer')

        with tf.name_scope('accuracy'):
            correct_pred = tf.equal(self.label, tf.round(sigmoid_pred))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

        self.train_writer = tf.summary.FileWriter(config.save_dir + '/' + 'train', sess.graph)
        self.valid_writer = tf.summary.FileWriter(config.save_dir + '/' + 'valid', sess.graph)
        self.test_writer = tf.summary.FileWriter(config.save_dir + '/' + 'test', sess.graph)
        self.optimize = optimize
        self.pred = sigmoid_pred
        self.accuracy = accuracy
        self.output_loss = cross_entropy_mean
        self.global_step = global_step
        self.saver = tf.train.Saver(max_to_keep=config.max_save)

        init = tf.global_variables_initializer()
        sess.run(init)

    def train_step(self, image_summary=False):
        summaries = []
        if image_summary:
            _, loss_batch, acc_batch, pred, x, y, file_id = \
                self.sess.run([self.optimize, self.output_loss, self.accuracy,
                               self.pred, self.input, self.label,
                               self.pipeline.file_id],
                               feed_dict={self.handle: self.train_handle,
                                          self.learning_rate: self.current_learning_rate,
                                          self.is_train: True,
                                          self.keep_prob: np.asarray(self.dropout)})
            img_x = x[:10, :, :].reshape((10*x.shape[1], x.shape[2])).T
            s = StringIO()
            plt.imsave(s, img_x, format='png')
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img_x.shape[0],
                                       width=img_x.shape[1])
            summaries.append(tf.Summary.Value(tag='train_{}/spec'.format(file_id[0]),
                                              image=img_sum))
            img_pred = pred[:10, :, :].reshape((10*pred.shape[1], pred.shape[2])).T
            s = StringIO()
            plt.imsave(s, img_pred, format='png')
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img_pred.shape[0],
                                       width=img_pred.shape[1])
            summaries.append(tf.Summary.Value(tag='train_{}/pred'.format(file_id[0]),
                                              image=img_sum))
            img_label = y[:10, :, :].reshape((10*y.shape[1], y.shape[2])).T
            s = StringIO()
            plt.imsave(s, img_label, format='png')
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img_label.shape[0],
                                       width=img_label.shape[1])
            summaries.append(tf.Summary.Value(tag='train_{}/label'.format(file_id[0]),
                                              image=img_sum))
        else:
            _, loss_batch, acc_batch = self.sess.run([self.optimize, self.output_loss, self.accuracy],
                                                     feed_dict={self.handle: self.train_handle,
                                                                self.learning_rate: self.current_learning_rate,
                                                                self.is_train: True,
                                                                self.keep_prob: np.asarray(self.dropout)})
        summaries.append(tf.Summary.Value(tag='accuracy',
                                          simple_value=acc_batch))
        summaries.append(tf.Summary.Value(tag='loss',
                                          simple_value=loss_batch))

        summary = tf.Summary(value=summaries)

        self.train_writer.add_summary(summary, self.sess.run(self.global_step))
        return loss_batch, acc_batch

    def validation(self):
        self.sess.run(self.pipeline.valid_iterator.initializer)
        t = time.time()
        n_valid = 0
        loss_valid = 0
        acc_valid = 0
        while True:
            try:
                loss_batch, acc_batch = self.sess.run([self.output_loss, self.accuracy],
                                                      feed_dict={self.handle: self.valid_handle,
                                                                 self.is_train: False,
                                                                 self.keep_prob: np.ones(len(self.dropout))})
                loss_valid += loss_batch
                acc_valid += acc_batch
                n_valid += 1
            except tf.errors.OutOfRangeError:
                break
        loss_valid /= n_valid
        acc_valid /= n_valid

        summaries = []
        summaries.append(tf.Summary.Value(tag='accuracy',
                                          simple_value=acc_valid))
        summaries.append(tf.Summary.Value(tag='loss',
                                          simple_value=loss_valid))
        summary = tf.Summary(value=summaries)

        self.valid_writer.add_summary(summary, self.sess.run(self.global_step))
        duration_valid = time.time() - t
        return loss_valid, acc_valid, duration_valid

    def evaluation(self):
        self.sess.run(self.pipeline.test_iterator.initializer)
        loss_test = acc_test = 0
        pred_list = []
        x_list = []
        y_list = []
        id_list = []
        n_test = 0
        t = time.time()
        while True:
            try:
                loss_batch, acc_batch, pred, x, y, file_id = self.sess.run([self.output_loss, self.accuracy,
                                                                   self.pred, self.input, self.label, self.pipeline.file_id],
                                                                  feed_dict={self.handle: self.test_handle,
                                                                             self.is_train: False,
                                                                             self.keep_prob: np.ones(
                                                                                 len(self.dropout))})
                loss_test += loss_batch
                acc_test += acc_batch
                n_test += 1
                pred = np.reshape(pred, (pred.shape[0] * pred.shape[1], pred.shape[2]))
                pred_list.append(pred)
                x = np.reshape(x, (x.shape[0] * x.shape[1], x.shape[2]))
                x_list.append(x)
                y = np.reshape(y, (y.shape[0] * y.shape[1], y.shape[2]))
                y_list.append(y)
                id_list.append(file_id[0])
            except tf.errors.OutOfRangeError:
                break
        loss_test /= n_test
        acc_test /= n_test

        summaries = []
        for n in xrange(10):
            img_x = x_list[n][:1000, :].T
            s = StringIO()
            plt.imsave(s, img_x, format='png')
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img_x.shape[0],
                                       width=img_x.shape[1])
            summaries.append(tf.Summary.Value(tag='test_{}/spec'.format(id_list[n]),
                                              image=img_sum))
            img_pred = pred_list[n][:1000, :].T
            s = StringIO()
            plt.imsave(s, img_pred, format='png')
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img_pred.shape[0],
                                       width=img_pred.shape[1])
            summaries.append(tf.Summary.Value(tag='test_{}/pred'.format(id_list[n]),
                                              image=img_sum))
            img_label = y_list[n][:1000, :].T
            s = StringIO()
            plt.imsave(s, img_label, format='png')
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img_label.shape[0],
                                       width=img_label.shape[1])
            summaries.append(tf.Summary.Value(tag='test_{}/label'.format(id_list[n]),
                                              image=img_sum))

        duration_test = time.time() - t
        pred = np.concatenate(pred_list, axis=0)
        int_pred = (pred >= 0.5).astype(np.int)

        y_test = np.concatenate(y_list, axis=0)
        precision, recall, accuracy, F = evaluate.myAccu(y_test, int_pred)
        summaries.append(tf.Summary.Value(tag='accuracy',
                                          simple_value=acc_test))
        summaries.append(tf.Summary.Value(tag='loss',
                                          simple_value=loss_test))
        summaries.append(tf.Summary.Value(tag='F-score',
                                          simple_value=F))
        summaries.append(tf.Summary.Value(tag='precision',
                                          simple_value=precision))
        summaries.append(tf.Summary.Value(tag='recall',
                                          simple_value=recall))
        summaries.append(tf.Summary.Value(tag='accuracy(eval)',
                                          simple_value=accuracy))
        summary = tf.Summary(value=summaries)
        self.test_writer.add_summary(summary, self.sess.run(self.global_step))
        return loss_test, acc_test, duration_test, n_test, precision, recall, accuracy, F, pred, int_pred, y_test

    def save_model(self):
        self.saver.save(self.sess, self.config.save_dir + '/' + "epoch", global_step=self.epoch)

    def restore_model(self):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.config.save_dir))

    def drop_learning_rate(self):
        self.n_drop += 1
        self.current_learning_rate *= self.config.lr_drop_multiplier
