import tensorflow as tf
import scipy.io as sio
import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt
import h5py




# hyper parameters
learning_rate = 0.0005
training_epochs = 30
batch_size = 100
valid_batch_size = 3000
trainSetRatio = 0.5
epsilon = 1e-3

# mat_contents= sio.loadmat('R5S2BdUbn5Gpre20aa5Uib')
mat_contents= h5py.File('R8dataS2Gpre20Ubn15UibId100Hb15postItr30_20_1_50_1_1000.mat', 'r')
# mat_contents= sio.loadmat('onsetCluster_chopin_smd')

vel_cel = mat_contents['velocityGainMatchingCell']
print('Refresh complete!!')
print(vel_cel.shape)
test_set_fold_cell = mat_contents[vel_cel[0, 0]]
print(test_set_fold_cell.shape)
# print(test)
spec_cell= mat_contents[test_set_fold_cell[3,0]]
print(spec_cell.shape)
print(spec_cell[0,100])
spec_array = mat_contents[spec_cell[0,100]]
print(spec_array.shape)
print(np.asarray(spec_array))
print(np.asarray(spec_array).any())
# print(spec_array)

print(len(vel_cel))
print(vel_cel[:])
print(vel_cel[0, 0])

print('Cell data length: ', vel_cel.shape)
test_contents = (sio.loadmat('onsetCluster_beethoven_ivory_scale'))


pieceSetSize = len( test_contents['onsetClusterArray'][0])



def unison_shuffled_copies(a,b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def loadPiece(fileName):
    test_contents = sio.loadmat(fileName)

    pieceX = []
    pieceY = []
    for i in range(pieceSetSize):
        # print(test_contents['onsetClusterArray'][0][i])
        tempX = np.asarray(test_contents['onsetClusterArray'][0][i]).reshape((445, 14, 1))
        pieceX.append(tempX / np.mean(tempX))
        # pieceY.append(np.asarray(test_contents['onsetMatchedVel'][0][i]).reshape(1))
        pieceY.append(velocityToOneHot(test_contents['onsetMatchedVel'][0][i]).reshape(13))

    pieceX = np.asarray(pieceX)
    pieceY = np.asarray(pieceY)
    return pieceX, pieceY


def velocityToOneHot(velocity):

    oneHotVec = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
    if velocity <20:
        oneHotVec[0] = 1
    elif velocity < 30:
        oneHotVec[1] = 1
    elif velocity < 40:
        oneHotVec[2] = 1
    elif velocity < 50:
        oneHotVec[3] = 1
    elif velocity < 55:
        oneHotVec[4] = 1
    elif velocity < 60:
        oneHotVec[5] = 1
    elif velocity < 65:
        oneHotVec[6] = 1
    elif velocity < 70:
        oneHotVec[7] = 1
    elif velocity < 80:
        oneHotVec[8] = 1
    elif velocity < 90:
        oneHotVec[9] = 1
    elif velocity < 100:
        oneHotVec[10] = 1
    elif velocity < 110:
        oneHotVec[11] = 1
    elif velocity < 128:
        oneHotVec[12] = 1

    return oneHotVec


# for i in range(trainSetSize):
dataSetX = []
dataSetY = []
trainSetX = []
trainSetY = []
testSetX = []
testSetY = []
pieceX = []
pieceY = []



for setIndex in range(1):#range(vel_cel.shape[1]):
    for foldIndex in range(1):#range(vel_cel.shape[0]):
        set_fold_cell = mat_contents[vel_cel[foldIndex, setIndex]]
        spec_cell = mat_contents[set_fold_cell[3, 0]]
        gt_vel_cell = mat_contents[set_fold_cell[2,0]]
        # spectrogramCell = np.asarray(vel_cel[foldIndex, setIndex][3,0])
        print(set_fold_cell.shape)
        # print('vel cell shape:', gt_vel_cell.shape)
        for j in range(spec_cell.shape[0]):
            print('Set:', setIndex, 'fold:',foldIndex,'pitch:',j ,'dataSize:',len(dataSetY))
            for dataSize in range (spec_cell.shape[1]):
                spec_array = np.transpose(np.asarray(mat_contents[spec_cell[j, dataSize]]))
                # print('spec_array shape:', spec_array.shape)
                if not spec_array.any():
                    break
            #     # dataSize += 1
            # print('dataSize:', dataSize)
            # for i in range(dataSize):
                # for targetPitchIndex in range(88):
                # if targetPitchIndex:
                tempX = spec_array.reshape((445, 14, 1))
                # tempX = spectrogramCell[i, j].reshape((445,14,1))
                dataSetX.append(tempX/np.mean(tempX))
                # dataSetX.append(tempX)
                # dataSetY.append(np.asarray(vel_cel[setIndex, foldIndex][0, 2][i, j].reshape(1)))
                # print('gt_Vel:' ,np.asarray(gt_vel_cell[j,dataSize]))


                # dataSetY.append(np.asarray(gt_vel_cell[j,dataSize]).reshape(1))
                dataSetY.append(velocityToOneHot(gt_vel_cell[j,dataSize]).reshape(13))


                # temp = np.asarray(cell[0][0][0,3][i, targetPitchIndex]).reshape((50))
                # if i < trainSetSize:
                    # trainSetX.append(temp)
                    # trainSetY.append(np.asarray(cell[0][0][0,2][i, targetPitchIndex].reshape(1)))
                # else:
                    # testSetX.append(temp)
                    # testSetY.append(np.asarray(cell[0][0][0,2][i, targetPitchIndex].reshape(1)))
                # print('dataSize:', len(dataSetX))




dataSetX = np.asarray(dataSetX)
dataSetY = np.asarray(dataSetY)
# pieceX = np.asarray(pieceX)
# pieceY = np.asarray(pieceY).reshape(pieceSetSize,1)

dataSetSize = dataSetX.shape[0]
trainSetSize = int(dataSetSize *trainSetRatio)


dataSetX, dataSetY = unison_shuffled_copies(dataSetX,dataSetY)

# trainSetX = np.asarray(trainSetX)
# trainSetY = np.asarray(trainSetY)
# testSetX = np.asarray(testSetX)
# testSetY = np.asarray(testSetY)
trainSetX = dataSetX[0:trainSetSize,:,:]
trainSetY = dataSetY[0:trainSetSize,:]
testSetX = dataSetX[trainSetSize:len(dataSetX),:,:]
testSetY = dataSetY[trainSetSize:len(dataSetY),:]

print(trainSetX.shape,  trainSetY.shape, testSetX.shape)


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



def build_graph():
    # input place holders
    X = tf.placeholder(tf.float32, [None, 445, 14, 1])  # img 28x28x1 (black/white)
    Y = tf.placeholder(tf.float32, [None, 13])
    is_training = tf.placeholder(tf.bool)
    # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
    # keep_prob = tf.placeholder(tf.float32)
    W1 = tf.get_variable("W1", shape=[5, 3, 1, 32],
                         initializer=tf.contrib.layers.xavier_initializer())

    # W1 = tf.Variable(tf.random_normal([5, 3, 1, 32], stddev=0.01))
    L1 = tf.nn.conv2d(X, W1, strides=[1, 2, 1, 1], padding='SAME')
    L1_flat = tf.reshape(L1, [-1, 223*14*32])
    print(L1)
    BN1_flat = batch_norm_wrapper(L1_flat, is_training=is_training)
    BN1 = tf.reshape(BN1_flat, [-1, 223,14,32])
    L1 = tf.nn.relu(BN1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 1, 1],
                        strides=[1, 2, 1, 1], padding='SAME')
    print(L1)




    W2 = tf.get_variable("W2", shape=[3, 3, 32, 8],
                         initializer=tf.contrib.layers.xavier_initializer())
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    print(L2)
    L2_flat = tf.reshape(L2, [-1, 112 * 14 * 4])
    BN2_flat = batch_norm_wrapper(L2_flat, is_training=is_training)
    BN2 = tf.reshape(BN2_flat, [-1, 112, 14, 4])
    L2 = tf.nn.relu(BN2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    print(L2)
    # L2_flat = tf.reshape(L2, [-1, 56 * 7 * 4])


    W3 = tf.get_variable("W3", shape=[5, 5, 4, 4],
                         initializer=tf.contrib.layers.xavier_initializer())
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    print(L3)
    L3_flat = tf.reshape(L3, [-1, 56 * 7 * 4])
    BN3_flat = batch_norm_wrapper(L3_flat, is_training=is_training)
    BN3 = tf.reshape(BN3_flat, [-1, 56, 7, 4])
    L3 = tf.nn.relu(BN3)
    # L3 = tf.nn.max_pool(L3, ksize=[1, 3, 3, 1],
    #                     strides=[1, 2, 2, 1], padding='SAME')
    print(L3)
    L3_flat = tf.reshape(L3, [-1, 56 * 7 * 4])



    W4 = tf.get_variable("W4", shape=[56 * 7 * 4, 13],
                         initializer=tf.contrib.layers.xavier_initializer())
    print(W4)
    b = tf.Variable(tf.random_normal([13]))
    hypothesis = tf.matmul(L3_flat, W4) + b

    print('hypothesis: ', hypothesis)
    print('Y: ', Y)

    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=hypothesis, labels=Y))
    # cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


    return (X, Y), optimizer, cost, hypothesis, tf.train.Saver(), is_training





# input place holders


# L1 ImgIn shape=(?, 28, 28, 1)


'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)

# initialize
sess = tf.Session()

sess.close()
tf.reset_default_graph()
(X, Y), optimizer, cost, hypothesis, saver,is_training = build_graph()

# train my model
# accuracy = tf.reduce_mean(1-tf.abs(hypothesis-Y)/Y)
# mean_error = tf.reduce_mean(tf.abs(hypothesis-Y))

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
pred2 = tf.reduce_mean(tf.cast(tf.less_equal(tf.abs(tf.argmax(hypothesis, 1) - tf.argmax(Y, 1)) , 1), tf.float32 ))
pred3 = tf.reduce_mean(tf.abs(tf.argmax(hypothesis, 1) - tf.argmax(Y, 1)))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
# train my model
    sess.run(tf.global_variables_initializer())
    print('Learning started. It takes sometime.')
    for epoch in range(training_epochs):
        avg_cost = 0
        avg_validAccu = 0
        avg_validCost = 0
        avg_validError = 0
        avg_validGuess = 0
        total_batch = int(trainSetSize / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = trainSetX[i * batch_size:(i + 1) * batch_size], trainSetY[
                                                                                 i * batch_size:(i + 1) * batch_size]
            feed_dict = {X: batch_xs, Y: batch_ys, is_training: True}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

        total_valid_batch = int(testSetX.shape[0] / valid_batch_size)
        for i in range(total_valid_batch):
            batch_xs, batch_ys = testSetX[i * valid_batch_size:(i + 1) * valid_batch_size], testSetY[
                                                                                 i * valid_batch_size:(i + 1) * valid_batch_size]
            feed_dict = {X: batch_xs, Y: batch_ys, is_training: False}
            # c, validAccu, validError = sess.run([cost, accuracy, mean_error], feed_dict=feed_dict)
            c, validAccu, validGuess = sess.run([cost, accuracy, pred2], feed_dict=feed_dict)

            avg_validCost += c / total_valid_batch
            avg_validAccu += validAccu / total_valid_batch
            # avg_validError += validError / total_valid_batch
            avg_validGuess += validGuess / total_valid_batch
        # validAccu = sess.run((accuracy, mean_error, cost), feed_dict={X: testSetX, Y: testSetY, is_training: False})
        print('Validation Accuracy:', avg_validAccu, 'Validation Cost:', avg_validCost, 'Validation Guess:', avg_validGuess)
    saved_model = saver.save(sess, './temp-bn-save')

    pieceX, pieceY = loadPiece('onsetCluster_beethoven_ivory_scale')
    pieceAccu,result =  sess.run([(accuracy), hypothesis], feed_dict={X: pieceX, Y: pieceY, is_training:False})
    # result =  sess.run(hypothesis, feed_dict={X: pieceX, Y: pieceY, is_training:False})
    (mu, sigma) = norm.fit(result)
    sigma = sigma * sqrt(2)
    print(mu, sigma)
    print('Piece Accuracy:', pieceAccu)

    pieceX, pieceY = loadPiece('onsetCluster_chopin_smd')
    pieceAccu, result = sess.run([(accuracy), hypothesis], feed_dict={X: pieceX, Y: pieceY, is_training: False})
    # result =  sess.run(hypothesis, feed_dict={X: pieceX, Y: pieceY, is_training:False})
    (mu, sigma) = norm.fit(result)
    sigma = sigma * sqrt(2)
    print(mu, sigma)
    print('Piece Accuracy:', pieceAccu)
print('Learning Finished!')
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