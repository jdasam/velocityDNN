from __future__ import division
import tensorflow as tf
import scipy.io as sio
import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from math import sqrt, ceil
import hdf5storage
import loadMatToNumpy as loadMat
import argparse
import os
import csv
import time


parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--sessMode", type=str, default='train', help="train or test")
# parser.add_argument("-sr", "--sample_rate", type=int, default=44100, help="sample rate")
# parser.add_argument("-f", "--n_fft", type=int, default=8192, help="fft_length")
# parser.add_argument("-res", "--resolution", type=int, default=4, help="number of bin per semitone")
# parser.add_argument("-hop", "--hop_size", type=int, default=441, help="hop_size")
# parser.add_argument("-d", "--del_dup", action='store_true', help="delete_duplicate_fft_bin")
# parser.add_argument("-low", "--lowest_bin", type=int, default=21, help="lower midi range")
# parser.add_argument("-high", "--highest_bin", type=int, default=108, help="upper midi range")
parser.add_argument("-model", "--nnModel", type=str, default="cnn", help="cnn or fcn")
parser.add_argument("-path", "--testPath", type=str, default="./testMat", help="folder path of test mat")
parser.add_argument("-tset", "--trainingSet", type=str, default="dataOneHot", help="training set folder path")
args = parser.parse_args()

# python ~.py -mode=test
# pos_weight_list = tf.constant([3.0, 1.5, 1.0, 0.5, 0.5, 1.0, 1.5, 3.0 ])
pos_weight_list = tf.constant([3.5, 2.0, 1.0, 0.4, 0.4, 1.0, 2.0, 3.5])


def _parse_function(example_proto):
    features = {
        "feature": tf.FixedLenFeature([445 * loadMat.specLength], tf.float32),
        # "label": tf.FixedLenFeature([loadMat.velClassNum], tf.float32)}
        "label": tf.FixedLenFeature(1, tf.float32)}
    ex = tf.parse_single_example(example_proto, features)

    label = ex["label"]
    feature = tf.reshape(ex['feature'], [445, loadMat.specLength])
    # def f1(): return tf.constant(1)
    # def f2(): return tf.constant(1)
    # pos_weight = tf.cond( tf.less_equal(tf.abs(tf.argmax(label,0)-3), 1), f1, f2 )
    # pos_weight = tf.case( [  (tf.equal(tf.argmax(label,0),0), f1)  ])
    pos_weight = tf.gather(pos_weight_list, tf.argmax(label,0))
    # pos_weight = tf.constant(1)

    # print(pos_weight)

    return feature, label, pos_weight

def one_shot_dataset(record_files, batch_size=128, buffer_size=1000, shuffle=True, num_threads=4):
    dataset = tf.contrib.data.TFRecordDataset(record_files)
    dataset = dataset.map(lambda x: _parse_function(x), num_threads=num_threads)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return dataset, iterator


def initalizable_dataset(record_files, batch_size=128, num_threads=4):
    dataset = tf.contrib.data.TFRecordDataset(record_files)
    dataset = dataset.map(lambda x: _parse_function(x), num_threads=num_threads)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    return dataset, iterator





# hyper parameters
learning_rate = 0.0001
training_epochs = 100
batch_size = 512
valid_batch_size = 512
trainSetRatio = 0.7
epsilon = 1e-7
mode = 'train'

# saver = tf.train.Saver(max_to_keep=3)

train_dataset, train_iterator = one_shot_dataset( args.trainingSet+'/train_.tfrecords',
                                                 batch_size, 1000)
_, valid_iterator = initalizable_dataset(args.trainingSet+'/valid_.tfrecords',
                                         valid_batch_size)
_, test_iterator = initalizable_dataset(args.trainingSet+'/test_.tfrecords',
                                        batch_size=1, num_threads=1)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.contrib.data.Iterator.from_string_handle(
    handle, train_dataset.output_types, train_dataset.output_shapes)
feature, label, pos_weight = iterator.get_next()



print('Refresh complete!!!!')
# test_set_fold_cell = mat_contents[vel_cel[0, 0]]
# print(test_set_fold_cell.shape)
# print(test)
# spec_cell= mat_contents[test_set_fold_cell[3,0]]
# print(spec_cell.shape)
# print(spec_cell[0,100])
# spec_array = mat_contents[spec_cell[0,100]]
# print(spec_array.shape)
# print(np.asarray(spec_array))
# print(np.asarray(spec_array).any())
# print(spec_array)
#
# print(len(vel_cel))
# print(vel_cel[:])
# print(vel_cel[0, 0])
# print('Cell data length: ', vel_cel.shape)




# for i in range(trainSetSize):

# pieceX, pieceY = loadMat.loadPiece('onsetCluster_beethoven_ivory_scale')
# print(pieceX, pieceY)



# print(pieceY)
# histogram = np.histogram(pieceY, bins=30, normed=True)
# print(histogram)
#
# (mu, sigma) = norm.fit(pieceY)
#
# print(mu, sigma)
#
# mean = np.mean(pieceY)
# std = np.std(pieceY)
# print((mean,std) )
# plt.plot(x, y)
# plt.show()


def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training is not None:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)


def conv_with_batch_norm(input_tensor, shape, scope_name, is_train, padding='SAME',
                        batch_norm=True, regularize=0, dilations=None):
   if dilations is None:
       dilations = [1, 1, 1, 1]

   with tf.variable_scope(scope_name) as scope:
       kernel = tf.get_variable('kernel', shape=shape,
                                initializer=tf.contrib.layers.xavier_initializer())
       conv = tf.nn.conv2d(input_tensor, kernel, dilations, padding=padding, name='conv')

       if batch_norm:
           conv = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay=0.9, epsilon=1e-07,
                                                     is_training=is_train)
       else:
           biases = tf.get_variable('biases', [shape[-1]], initializer=tf.constant_initializer(0.0))
           conv = tf.nn.bias_add(conv, biases)
       if regularize > 0:
           tf.add_to_collection('losses', tf.multiply(tf.nn.l2_loss(kernel), regularize))

       activation = tf.nn.relu(conv, name='activation')

   return activation



def build_graph(feature, label, pos_weight):
    # input place holders
    # X = tf.placeholder(tf.float32, [None, 445, 14, 1])  # img 445 * 14 *1
    # Y = tf.placeholder(tf.float32, [None, 13])

    Y = label
    # print (Y)
    print('label: ', Y[0,:])
    # Y = loadMat.iteratorToOneHot(label)

    is_training = tf.placeholder(tf.bool)




    if args.nnModel == "cnn":
        X = tf.expand_dims(feature, -1, name='X')
    # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
    # keep_prob = tf.placeholder(tf.float32)
        W1 = tf.get_variable("W1", shape=[5, 3, 1, 32],
                             initializer=tf.contrib.layers.xavier_initializer())

        # W1 = tf.Variable(tf.random_normal([5, 3, 1, 32], stddev=0.01))
        L1 = tf.nn.conv2d(X, W1, strides=[1, 2, 1, 1], padding='SAME')
        L1_flat = tf.reshape(L1, [-1, 223*loadMat.specLength*32])
        print(L1)
        BN1_flat = batch_norm_wrapper(L1_flat, is_training=is_training)
        BN1 = tf.reshape(BN1_flat, [-1, 223,loadMat.specLength,32])
        L1 = tf.nn.relu(BN1)
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 1, 1],
                            strides=[1, 2, 1, 1], padding='SAME')
        print(L1)




        W2 = tf.get_variable("W2", shape=[3, 3, 32, 8],
                             initializer=tf.contrib.layers.xavier_initializer())
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        print(L2)
        L2_flat = tf.reshape(L2, [-1, 112 * loadMat.specLength * 8])
        BN2_flat = batch_norm_wrapper(L2_flat, is_training=is_training)
        BN2 = tf.reshape(BN2_flat, [-1, 112, loadMat.specLength, 8])
        L2 = tf.nn.relu(BN2)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        print(L2)
        # L2_flat = tf.reshape(L2, [-1, 56 * 7 * 4])


        W3 = tf.get_variable("W3", shape=[5, 5, 8, 4],
                             initializer=tf.contrib.layers.xavier_initializer())
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        print(L3)
        print L3.get_shape()
        L3_flat = tf.reshape(L3, [-1, 56 * 7 * 4])
        BN3_flat = batch_norm_wrapper(L3_flat, is_training=is_training)
        BN3 = tf.reshape(BN3_flat, [-1, 56, 7, 4])
        L3 = tf.nn.relu(BN3)
        L3 = tf.nn.max_pool(L3, ksize=[1, 3, 3, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
        print(L3)
        L3_flat = tf.reshape(L3, [-1, 28 * 4 * 4])



        W4 = tf.get_variable("W4", shape=[28 * 4 * 4, 300],
                             initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.Variable(tf.random_normal([300]))
        L4 = tf.matmul(L3_flat, W4) + b4
        BN4_flat = batch_norm_wrapper(L4, is_training=is_training)
        BN4 = tf.reshape(BN4_flat, [-1, 300])
        L4 = tf.nn.relu(BN4)



        W5 = tf.get_variable("W5", shape=[300, loadMat.velClassNum],
                             initializer=tf.contrib.layers.xavier_initializer())
        print(W5)
        b = tf.Variable(tf.random_normal([loadMat.velClassNum]))
        hypothesis = tf.matmul(L4, W5) + b

        print('hypothesis: ', hypothesis)
        print('Y: ', Y)
    elif args.nnModel == "fcn":
        reg = 0.0001
        X = tf.reshape(feature, [-1,445*loadMat.specLength])
        Fc1 = tf.contrib.layers.fully_connected(inputs=X, num_outputs=256, activation_fn=tf.nn.selu, weights_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))
        Fc2 = tf.contrib.layers.fully_connected(inputs=Fc1, num_outputs=256, activation_fn=tf.nn.selu, weights_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))
        Fc3 = tf.contrib.layers.fully_connected(inputs=Fc2, num_outputs=256, activation_fn=tf.nn.selu, weights_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))
        Fc4 = tf.contrib.layers.fully_connected(inputs=Fc3, num_outputs=256, activation_fn=tf.nn.selu, weights_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))
        Fc5 = tf.contrib.layers.fully_connected(inputs=Fc4, num_outputs=256, activation_fn=tf.nn.selu, weights_regularizer = tf.contrib.layers.l2_regularizer(scale=reg))
        # hypothesis = tf.contrib.layers.fully_connected(inputs=Fc5, num_outputs=loadMat.velClassNum, activation_fn=tf.nn.relu)
        hypothesis = tf.contrib.layers.fully_connected(inputs=Fc5, num_outputs=1, activation_fn=tf.nn.relu)

        print('hypotht: ',hypothesis)

    elif args.nnModel =='single':
        X = tf.reshape(feature, [-1,445*loadMat.specLength])
        hypothesis = tf.contrib.layers.fully_connected(inputs=X, num_outputs=loadMat.velClassNum, activation_fn=tf.nn.relu)


    # define cost/loss & optimizer
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = hypothesis, labels=Y)
    # def f1(): return tf.constant(0.5)
    # def f2(): return tf.constant(1.5)
    # test = tf.constant(5)
    # print('argmax works', tf.argmax(Y,1))
    # # pos_weight = tf.cond( tf.less_equal(tf.abs(tf.argmax(Y,1)-6), 3), f1, f2 )
    # pos_weight = tf.cond(tf.less_equal(tf.argmax(Y,1), 3), f1, f2)
    # print('Pos_weight: ', pos_weight)


    weighted_cross_entropy = tf.losses.compute_weighted_loss(cross_entropy, pos_weight)
    # cost = tf.reduce_mean(weighted_cross_entropy)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        # logits=hypothesis, labels=Y))
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    return (X, Y), optimizer, cost, hypothesis, tf.train.Saver(max_to_keep=1), is_training





# input place holders


# L1 ImgIn shape=(?, 28, 28, 1)


'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)

# initialize
(X, Y), optimizer, cost, hypothesis, saver,is_training = build_graph(feature, label, pos_weight)

# train my model
# accuracy = tf.reduce_mean(1-tf.abs(hypothesis-Y)/Y)
# mean_error = tf.reduce_mean(tf.abs(hypothesis-Y))

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
pred2 = tf.reduce_mean(tf.cast(tf.less_equal(tf.abs(tf.argmax(hypothesis, 1) - tf.argmax(Y, 1)) , 1), tf.float32 ))
pred3 = tf.reduce_mean(tf.abs(tf.argmax(hypothesis, 1) - tf.argmax(Y, 1)))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accuracy = tf.reduce_mean(tf.cast(tf.less_equal(tf.abs(Y-hypothesis), 5),tf.float32 ) )


# c = tf.ConfigProto()
# c.gpu_options.visible_device_list = str(1)
# c.gpu_options.allow_growth = True

if args.sessMode  == 'train':
    with tf.Session() as sess:
    # train my model
        sess.run(tf.global_variables_initializer())
        # dataSetX, dataSetY = loadMat.loadTrainSet('R8dataS2Gpre20Ubn15UibId100Hb15postItr30_20_1_50_1_1000.mat')
        # dataSetSize = dataSetX.shape[0]
        # trainSetSize = int(dataSetSize *trainSetRatio)
        #
        # dataSetX, dataSetY = loadMat.unison_shuffled_copies(dataSetX,dataSetY)
        # trainSetX, trainSetY, testSetX, testSetY = loadMat.dataSetToTrainTest(dataSetX, dataSetY, trainSetSize)
        train_handle = sess.run(train_iterator.string_handle())
        valid_handle = sess.run(valid_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        former_validation_cost = np.inf
        stop_patience = 0
        patience_limit = 5


        print('Learning started. It takes sometime.')
        for epoch in range(training_epochs):
            avg_cost = 0
            avg_validAccu = 0
            avg_validCost = 0
            avg_validError = 0
            avg_validGuess = 0
            # total_batch = int(207817 *0.7 / batch_size)
            total_batch = int(97021 *0.7 / batch_size)
            # total_batch = int(95920 *0.8 / batch_size)
            # total_batch = int(194042 * 0.8 / batch_size)
            t= time.time()

            for i in range(total_batch):
            #     batch_xs, batch_ys = trainSetX[i * batch_size:(i + 1) * batch_size], trainSetY[
            #                                                                          i * batch_size:(i + 1) * batch_size]
            #     feed_dict = {X: batch_xs, Y: batch_ys, is_training: True}
            #     c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            #     avg_cost += c / total_batch
                c, _ = sess.run([cost, optimizer], feed_dict={handle: train_handle,
                                                                is_training: True,
                                                                })
                avg_cost += c
            avg_cost /=  total_batch
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

            # total_valid_batch = int( 287490 *0.2 / valid_batch_size)
            sess.run(valid_iterator.initializer)
            total_valid_batch = 0
            while True:
                total_valid_batch += 1
                try:

                    # for i in range(total_valid_batch):
                    # batch_xs, batch_ys = testSetX[i * valid_batch_size:(i + 1) * valid_batch_size], testSetY[
                    #                                                                      i * valid_batch_size:(i + 1) * valid_batch_size]
                    feed_dict = {handle: valid_handle, is_training: False}
                    # c, validAccu, validError = sess.run([cost, accuracy, mean_error], feed_dict=feed_dict)
                    c, validAccu, validGuess = sess.run([cost, accuracy, pred2], feed_dict=feed_dict)

                    avg_validCost += c
                    avg_validAccu += validAccu
                    # avg_validError += validError / total_valid_batch
                    avg_validGuess += validGuess
                except tf.errors.OutOfRangeError:
                    print('count_end. time={:f}, n_batch={:d}'.format(time.time() - t, total_batch))
                    break
            avg_validAccu /= total_valid_batch
            avg_validCost /= total_valid_batch
            avg_validGuess/= total_valid_batch
            # validAccu = sess.run((accuracy, mean_error, cost), feed_dict={X: testSetX, Y: testSetY, is_training: False})
            print('Validation Accuracy:', avg_validAccu, 'Validation Cost:', avg_validCost, 'Validation Guess:', avg_validGuess)
            saved_model = saver.save(sess, './savedModel_'+args.trainingSet+'/temp-bn-save')


            if epoch > training_epochs/10:
                if former_validation_cost < avg_validCost:
                    stop_patience += 1

                    if stop_patience == patience_limit:
                        print("Early Stopping Worked")
                        break
                else:
                    stop_patience=0
                    former_validation_cost = avg_validCost




        print('Learning Finished!')
else:
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('./savedModel_'+args.trainingSet+'/'))
        # fileList = os.listdir(args.testPath)
        fileList = loadMat.readExtInFolder(args.testPath, 'mat')
        totalStat = []
        print('File List: ', fileList)
        for pieceIndex in range(len(fileList)):
            testMatName = args.testPath+'/'+fileList[pieceIndex].split('.mat')[0]
            pieceX, pieceY = loadMat.loadPiece(testMatName)
            test_batch = int(ceil(pieceX.shape[0]/valid_batch_size))
            # print('number of test batch: ',test_batch)
            result = np.empty([0, loadMat.velClassNum])
            avg_pieceAccu = 0
            for i in range(test_batch):
                batch_xs, batch_ys = pieceX[i * valid_batch_size:(i + 1) * valid_batch_size], pieceY[i * valid_batch_size:(i + 1) * valid_batch_size]
                batch_xs = batch_xs.reshape([-1, 445*loadMat.specLength])
                feed_dict = {X: batch_xs, Y: batch_ys, is_training: False}
                # c, validAccu, validError = sess.run([cost, accuracy, mean_error], feed_dict=feed_dict)
                pieceAccu, tempResult = sess.run([accuracy, hypothesis], feed_dict=feed_dict)
                avg_pieceAccu += pieceAccu
                result = np.concatenate((result, tempResult), axis=0)
                # result = result.flatten()
            pieceAccu = avg_pieceAccu / test_batch
            # pieceAccu,result =  sess.run([(accuracy), hypothesis], feed_dict={X: pieceX, Y: pieceY, is_training:False})
            # result =  sess.run(hypothesis, feed_dict={X: pieceX, Y: pieceY, is_training:False})

            # print("Result Value: ", result.shape)
            # velocity = loadMat.resultToVelocity(result)
            velocity = result
            mu, sigma = loadMat.velocityToStatistic(velocity)
            # print(velocity)

            sigma = sigma * sqrt(2)
            # vel_error = loadMat.calError(pieceY, result)
            vel_error = np.mean(np.abs(pieceY-result))
            print('Piece name is ', testMatName, 'and statistics are ', (mu, sigma))
            print('Piece Accuracy:', pieceAccu, 'and mean velocity error is', vel_error)

            csv_name = fileList[pieceIndex].split('.mp3')[0] + '.csv'
            csv_dir = args.testPath + '/'
            csv_file = open(csv_dir+csv_name, 'w')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(velocity)
            csv_file.close()
            totalStat.append(mu)
            totalStat.append(sigma)
        csv_name ='totalStat.csv'
        csv_dir = args.testPath + '/'
        csv_file = open(csv_dir + csv_name, 'w')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(totalStat)
        csv_file.close()

# print(result)


# tf.reset_default_graph()
# (X, Y), _, cost, hypothesis, saver = build_graph(is_training=False)


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver.restore(sess, './temp-bn-save')
#     validAccu = sess.run((accuracy, mean_error), feed_dict={X: testSetX, Y: testSetY})
#     pieceAccu =  sess.run((accuracy,mean_error), feed_dict={X: pieceX, Y: pieceY})
#     # accu = sess.run((accuracy, mean_error, hypothesis), feed_dict={X: pieceX, Y: pieceY})
#     print('Validation Accuracy:', validAccu, 'Piece Accuracy',pieceAccu)


# plt.hist(pieceAccu, bins=30, normed=True)
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# y = norm.pdf(x, mean, std)
# plt.plot(x, y)
# plt.show()


