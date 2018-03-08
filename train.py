from __future__ import division

import tensorflow as tf
import numpy as np
import time
import datetime
import os
import model
import utils
from make_records import _parse_function
# from make_records_chroma import _parse_function
from six.moves import xrange


# tf.set_random_seed(1257)

'''
def _parse_function(example_proto):
    features = {"spec": tf.FixedLenFeature([3900], tf.float32),
                "mask": tf.FixedLenFeature([4400], tf.float32),
                "label": tf.FixedLenFeature([4400], tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    input = tf.reshape(parsed_features["spec"], [100, -1])
    label = tf.reshape(parsed_features["label"], [100, -1])
    mask = tf.reshape(parsed_features["mask"], [100, -1])
    # label = label[2:-2, :]
    # mask = mask[2:-2, :]
    # onset = tf.where(tf.equal(label, 2))
    # label = tf.cast(tf.reshape(tf.clip_by_value(parsed_features["label"], 0, 1), [100, -1]), tf.float32)

    return input, label, mask
'''


def one_shot_dataset(record_files, batch_size=128, buffer_size=20, shuffle=True, num_threads=4):
    dataset = tf.contrib.data.TFRecordDataset(record_files)
    dataset = dataset.map(_parse_function, num_threads=num_threads)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return dataset, iterator


def initalizable_dataset(record_files, batch_size=128, buffer_size=20, shuffle=False, num_threads=4):
    dataset = tf.contrib.data.TFRecordDataset(record_files)
    dataset = dataset.map(_parse_function, num_threads=num_threads)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return dataset, iterator


class InputPipeline:
    def __init__(self, config):
        self.config = config
        train_files = [os.path.join(config.data_dir, el) for el in os.listdir(config.data_dir) if 'train' in el]
        valid_files = [os.path.join(config.data_dir, el) for el in os.listdir(config.data_dir) if 'valid' in el]
        test_files = [os.path.join(config.data_dir, el) for el in os.listdir(config.data_dir) if 'test' in el]

        # train_dataset, train_iterator = one_shot_dataset(train_files, config.batch_size, config.buffer_size)
        train_dataset, train_iterator = initalizable_dataset(train_files, config.batch_size, config.buffer_size, True)
        _, valid_iterator = initalizable_dataset(valid_files, config.batch_size)
        _, test_iterator = initalizable_dataset(test_files, config.batch_size)

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.contrib.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)
        input, label, mask, file_id = iterator.get_next()
        label = tf.cast(label, tf.float32)

        self.file_id = file_id
        self.input_raw = input
        self.label_raw = label
        self.mask_raw = mask
        self.handle = handle
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.test_iterator = test_iterator
        self.input = input
        self.label = tf.clip_by_value(label, 0, 1)
        self.mask = None
        self.train_dataset = train_dataset


    def modify(self):
        if self.config.add_diff:
            # TODO: THIS is Wrong. diff[0,:] is not right
            diff = tf.concat([self.input_raw[0:1, :], self.input_raw[1:, :] - self.input_raw[:-1, :]], axis=0)
            self.input = tf.concat([self.input_raw, diff], axis=-1)
        if self.config.chroma:
            assert self.config.output_dim == 12
            # TODO: chroma with tf
            pass
        if self.config.onset:
            # TODO: add onset label
            pass
        if self.config.onset_weight:
            if self.config.onset_weight == 'type1':
                self.mask = self.mask_raw * 20 + 1


def check_improve(current, best, monitor):
    if monitor == 'acc':
        return current > best
    else:
        return current < best


def count_batches(sess, record_files, batch_size):
    print('count_start')
    t = time.time()
    _, iterator = initalizable_dataset(record_files, batch_size, num_threads=4)
    sess.run(iterator.initializer)
    n_batch = 0
    while True:
        try:
            _, _, _, _ = sess.run(iterator.get_next())
            n_batch += 1
        except tf.errors.OutOfRangeError:
            print('count_end. time={:f}, n_batch={:d}'.format(time.time()-t, n_batch))
            break
    return n_batch


class Callback:
    def __init__(self, sess, config, model):
        self.sess = sess
        self.config = config
        self.model = model
        self.wait_after_best = 0
        if config.monitor == 'acc':
            self.monitor_best = -np.inf
        elif config.monitor == 'loss':
            self.monitor_best = np.inf
        self.monitor = None
        self.improved = None

    def train_start(self):
        utils.save_config(self.config)
        text_summary = 'run starts' + '\n' + str(datetime.datetime.now()) + '\n'
        print('save dir: {}'.format(self.config.save_dir))
        if self.config.verbose >= 1:
            print(text_summary)
        utils.record_as_text(self.config, text_summary)

    def validation_end(self, duration_valid, loss_valid, acc_valid):
        text_summary = "epoch {:d}: valid time = {:06.1f} sec, " \
            .format(self.model.epoch, duration_valid)
        text_summary += "[valid Loss/ valid Acc] : {:05.4f} / {:05.4f}". \
            format(loss_valid, acc_valid)

        if self.config.monitor == 'acc':
            self.monitor = acc_valid
        elif self.config.monitor == 'loss':
            self.monitor = loss_valid
        # TODO: count patients factor after every validation. -> every epoch
        self.improved, stop = self.early_stopping()
        if self.improved:
            text_summary += ' *'
        if self.config.verbose >= 2:
            print(text_summary)
        utils.record_as_text(self.config, text_summary + '\n')
        return stop

    def epoch_end(self, duration_train, loss_train, acc_train, loss_valid, acc_valid):
        text_summary = "epoch {:d}: train time = {:06.1f} sec, " \
            .format(self.model.epoch, duration_train)
        text_summary += "[valid Loss/ valid Acc] : {:05.4f} / {:05.4f}, [Loss/ Acc] {:05.4f} / {:05.4f}". \
            format(loss_valid, acc_valid, loss_train, acc_train)
        if self.improved:
            text_summary += ' *'
        if self.config.verbose >= 1:
            print(text_summary)
        utils.record_as_text(self.config, text_summary + '\n')

    def evaluation_end(self, duration_test, n_test, loss_test, acc_test, precision, recall, accuracy, F):
        text_summary = "Evaluation : test time = {:06.1f} sec / {:d} batches\n" \
            .format(duration_test, n_test)
        text_summary += "[Loss / Acc ] {:05.4f} / {:05.4f}\n" \
            .format(loss_test, acc_test)
        text_summary += "P: {:05.4f} / R: {:05.4f} / A: {:05.4f} / F: {:05.4f}"\
            .format(precision, recall, accuracy, F)
        if self.config.verbose >= 1:
            print(text_summary)
        utils.record_as_text(self.config, text_summary + '\n')

    def early_stopping(self):
        early_stop = False
        if check_improve(current=self.monitor, best=self.monitor_best, monitor=self.config.monitor):
            self.monitor_best = self.monitor
            self.wait_after_best = 0
            self.model.save_model()
            improved = True
        else:
            self.wait_after_best += 1
            improved = False

            if self.wait_after_best > self.config.patients:
                self.wait_after_best = 0
                self.model.drop_learning_rate()
                self.model.restore_model()
                text_summary = 'Learning Rate Drop: lr={:.3e}\n'.format(self.model.current_learning_rate)
                utils.record_as_text(self.config, text_summary)
                if self.config.verbose >= 1:
                    print(text_summary)
                early_stop = True

        return improved, early_stop


def main(config):
    c = tf.ConfigProto()
    c.gpu_options.visible_device_list = str(config.visible_gpu)
    # c.allow_soft_placement = True
    # c.gpu_options.allow_growth = True
    with tf.Session(config=c) as sess:
        pipeline = InputPipeline(config=config)
        pipeline.modify()

        my_model = model.Model(sess=sess, config=config, pipeline=pipeline)
        callback = Callback(sess=sess, config=config, model=my_model)
        callback.train_start()

        train_files = [os.path.join(config.data_dir, el) for el in os.listdir(config.data_dir) if 'train' in el]
        n_batch_train = count_batches(sess, train_files, config.batch_size)
        print n_batch_train
        # n_batch_train = 335  # 128 batch
        if config.debug:
            n_batch_train = 2
        while my_model.n_drop <= config.n_lr_drop:
            my_model.epoch += 1
            t_train = time.time()
            loss_train = 0
            acc_train = 0
            sess.run(pipeline.train_iterator.initializer)
            for train_steps in xrange(n_batch_train):
                if train_steps == 0:
                    loss_batch, acc_batch = my_model.train_step(image_summary=True)
                else:
                    loss_batch, acc_batch = my_model.train_step()
                loss_train += loss_batch
                acc_train += acc_batch
                n_validation = config.validation_per_epoch
                validation_points = [(el + 1) * n_batch_train // n_validation for el in range(n_validation)]
                if train_steps + 1 in validation_points:
                    loss_valid, acc_valid, duration_valid = my_model.validation()
                    early_stop = callback.validation_end(duration_valid, acc_valid=acc_valid, loss_valid=loss_valid)
                    if early_stop:
                        break
            loss_train /= (train_steps + 1)
            acc_train /= (train_steps + 1)
            duration_train = time.time() - t_train
            callback.epoch_end(duration_train, loss_train, acc_train, loss_valid, acc_valid)

        my_model.restore_model()

        loss_test, acc_test, duration_test, n_test, precision, recall, accuracy, F, pred, int_pred, y_test \
            = my_model.evaluation()
        callback.evaluation_end(duration_test, n_test, loss_test, acc_test, precision, recall, accuracy, F)

        sess.close()
