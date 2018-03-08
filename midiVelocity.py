import scipy.io as sio
import numpy as np
import tensorflow as tf
import random

mat_contents= ( sio.loadmat('R5S2bExt0ubn5Gpr20') )
cell = mat_contents['resultData']['velocityGainMatchingData']

test_contents = (sio.loadmat('onsetCluster'))


targetPitchIndex = 35

dataSetSize = 1188
trainSetSize = 1100

pieceSetSize = len( test_contents['onsetClusterArray'][0])

# parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 50
epsilon = 1e-3


def unison_shuffled_copies(a,b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



# for i in range(trainSetSize):
dataSetX = []
dataSetY = []
trainSetX = []
trainSetY = []
testSetX = []
testSetY = []
pieceX = []
pieceY = []

for i in range(dataSetSize):
    # for targetPitchIndex in range(88):
    # if targetPitchIndex:
    tempX = np.asarray(cell[0][0][0,3][i, targetPitchIndex]).reshape((50))
    dataSetX.append(tempX/np.mean(tempX))
    dataSetY.append(np.asarray(cell[0][0][0,2][i, targetPitchIndex].reshape(1)))
    # temp = np.asarray(cell[0][0][0,3][i, targetPitchIndex]).reshape((50))
    # if i < trainSetSize:
        # trainSetX.append(temp)
        # trainSetY.append(np.asarray(cell[0][0][0,2][i, targetPitchIndex].reshape(1)))
    # else:
        # testSetX.append(temp)
        # testSetY.append(np.asarray(cell[0][0][0,2][i, targetPitchIndex].reshape(1)))

print(len( test_contents['onsetClusterArray'][0]))

for i in range(pieceSetSize):
    print(test_contents['onsetClusterArray'][0][i])
    pieceX.append(np.asarray(test_contents['onsetClusterArray'][0][i]).reshape(50) )
    pieceY.append(np.asarray(test_contents['onsetMatchedVel'][0][i]))

dataSetX = np.asarray(dataSetX)
dataSetY = np.asarray(dataSetY)
pieceX = np.asarray(pieceX)
pieceY = np.asarray(pieceY).reshape(pieceSetSize,1)


dataSetX, dataSetY = unison_shuffled_copies(dataSetX,dataSetY)

# trainSetX = np.asarray(trainSetX)
# trainSetY = np.asarray(trainSetY)
# testSetX = np.asarray(testSetX)
# testSetY = np.asarray(testSetY)
trainSetX = dataSetX[0:trainSetSize,:]
trainSetY = dataSetY[0:trainSetSize,:]
testSetX = dataSetX[trainSetSize:len(dataSetX),:]
testSetY = dataSetY[trainSetSize:len(dataSetY),:]


print(trainSetX.shape,  trainSetY.shape, testSetX.shape, pieceY.shape)
# print(pieceY)


# trainSetX= np.append(cell[0][0][0,3][0:trainSetSize, targetPitchIndex], axis=0)
# trainSetY[i] = np.asarray(cell[0][0][0,2][0:trainSetSize, targetPitchIndex])


    # testSetX = np.asarray(cell[0][0][0,3][trainSetSize:len(cell[0][0][0,3]), targetPitchIndex])
    # testSetY = np.asarray(cell[0][0][0,2][trainSetSize:len(cell[0][0][0,3]), targetPitchIndex])



# print(trainSetX.shape)



tf.set_random_seed(777)  # reproducibility



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



def build_graph(is_training):
    # input place holders
    X = tf.placeholder(tf.float32, [None, 50])
    Y = tf.placeholder(tf.float32, [None, 1])

    # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
    # keep_prob = tf.placeholder(tf.float32)

    # weights & bias for nn layers
    # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    W1 = tf.get_variable("W1", shape=[50, 40],
                         initializer=tf.contrib.layers.xavier_initializer())
    Z1_BN = tf.matmul(X,W1)
    BN1 = batch_norm_wrapper(Z1_BN, is_training=is_training)
    L1= tf.nn.relu(BN1)
    # b1 = tf.Variable(tf.random_normal([20]))
    # L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    # L1 = tf.nn.dropout(L1, keep_prob=keep_prob)


    W2 = tf.get_variable("W2", shape=[40, 40],
                         initializer=tf.contrib.layers.xavier_initializer())
    # b2 = tf.Variable(tf.random_normal([20]))
    # L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    # L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    Z2 = tf.matmul(L1,W2)
    BN2 = batch_norm_wrapper(Z2, is_training=is_training)
    L2 = tf.nn.relu(BN2)

    W3 = tf.get_variable("W3", shape=[40, 40],
                         initializer=tf.contrib.layers.xavier_initializer())
    Z3 = tf.matmul(L2,W3)
    BN3 = batch_norm_wrapper(Z3, is_training=is_training)
    L3 = tf.nn.relu(BN3)


    W4 = tf.get_variable("W4", shape=[40, 1],
                         initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([1]))
    hypothesis = tf.matmul(L3, W4) + b4

    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    return (X, Y), optimizer, cost, hypothesis, tf.train.Saver()


# initialize
sess = tf.Session()
# sess.run(tf.global_variables_initializer())

sess.close()
tf.reset_default_graph()
(X, Y), optimizer, cost, _, saver = build_graph(is_training=True)

with tf.Session() as sess:
# train my model
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(trainSetSize / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = trainSetX[i*batch_size:(i+1)*batch_size], trainSetY[i*batch_size:(i+1)*batch_size]
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    saved_model = saver.save(sess, './temp-bn-save' )

print('Learning Finished!')


tf.reset_default_graph()
(X, Y), _, cost, hypothesis, saver = build_graph(is_training=False)

accuracy = tf.reduce_mean(1-tf.abs(hypothesis-Y)/Y)
mean_error = tf.reduce_mean(tf.abs(hypothesis-Y))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './temp-bn-save')
    accu = sess.run((accuracy, mean_error), feed_dict={X: testSetX, Y: testSetY})
    # accu = sess.run((accuracy, mean_error, hypothesis), feed_dict={X: pieceX, Y: pieceY})
    print('Accuracy:', accu)


# correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))



# accuracy = tf.reduce_mean(1-tf.abs(hypothesis-Y)/Y)
# print('Accuracy:', sess.run(accuracy, feed_dict={
#       X: testSetX, Y: testSetY, keep_prob: 1}))

# Get one and predict
# r = random.randint(0, mnist.test.num_examples - 1)
# print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction: ", sess.run(
#     tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))
