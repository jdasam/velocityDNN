# Lab 11 MNIST and Convolutional Neural Network
import tensorflow as tf
import scipy.io as sio
import numpy as np
import random
# import matplotlib.pyplot as plt





# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
trainSetRatio = 0.9

mat_contents= sio.loadmat('BachOnsetSpectrogram')
cell = mat_contents['velocityGainMatchingCell']
spectrogramCell = np.asarray(cell[0][0][0,3])

test_contents = (sio.loadmat('onsetCluster_ivory'))


pieceSetSize = len( test_contents['onsetClusterArray'][0])



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

for j in range(spectrogramCell.shape[1]):
    for dataSize in range (spectrogramCell.shape[0]):
        if not spectrogramCell[dataSize,j].any():
            break
        dataSize += 1
    for i in range(dataSize):
        # for targetPitchIndex in range(88):
        # if targetPitchIndex:
        tempX = np.asarray(cell[0][0][0,3][i, j]).reshape((445,14,1))
        dataSetX.append(tempX/np.mean(tempX))
        # dataSetX.append(tempX)
        dataSetY.append(np.asarray(cell[0][0][0,2][i, j].reshape(1)))
        # temp = np.asarray(cell[0][0][0,3][i, targetPitchIndex]).reshape((50))
        # if i < trainSetSize:
            # trainSetX.append(temp)
            # trainSetY.append(np.asarray(cell[0][0][0,2][i, targetPitchIndex].reshape(1)))
        # else:
            # testSetX.append(temp)
            # testSetY.append(np.asarray(cell[0][0][0,2][i, targetPitchIndex].reshape(1)))



for i in range(pieceSetSize):
    # print(test_contents['onsetClusterArray'][0][i])
    tempX = np.asarray(test_contents['onsetClusterArray'][0][i]).reshape((445,14,1))
    pieceX.append(tempX / np.mean(tempX))
    pieceY.append(np.asarray(test_contents['onsetMatchedVel'][0][i]))

dataSetX = np.asarray(dataSetX)
dataSetY = np.asarray(dataSetY)
pieceX = np.asarray(pieceX)
pieceY = np.asarray(pieceY).reshape(pieceSetSize,1)

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

print(trainSetX.shape,  trainSetY.shape, testSetX.shape, pieceY.shape)

# input place holders
X_img = tf.placeholder(tf.float32, [None, 445, 14, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 1])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([5, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 2, 1, 1], padding='SAME')
print(L1)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 1, 1],
                    strides=[1, 2, 1, 1], padding='SAME')
print(L1)

'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
print(L2)
L2_flat = tf.reshape(L2, [-1, 56 * 7 * 64])

'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
'''

# W3 = tf.Variable(tf.random_normal([3, 2, 64, 16], stddev=0.01))
# #    Conv      ->(?, 14, 14, 64)
# #    Pool      ->(?, 7, 7, 64)
# L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
# L3 = tf.nn.relu(L3)
# L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1],
#                     strides=[1, 2, 2, 1], padding='SAME')
# print(L3)
#

# Final FC 7x7x64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[56 * 7 * 64, 1],
                     initializer=tf.contrib.layers.xavier_initializer())
print(W3)
b = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(L2_flat, W3) + b

# define cost/loss & optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(trainSetSize / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = trainSetX[i * batch_size:(i + 1) * batch_size], trainSetY[
                                                                             i * batch_size:(i + 1) * batch_size]
        feed_dict = {X_img: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy = tf.reduce_mean(1-tf.abs(hypothesis-Y)/Y)
mean_error = tf.reduce_mean(tf.abs(hypothesis-Y))

print('Validation Accuracy:', sess.run((accuracy,mean_error), feed_dict={
      X_img: testSetX, Y: testSetY}))

print('Accuracy:', sess.run((accuracy,mean_error), feed_dict={
      X_img: pieceX, Y: pieceY}))



# Get one and predict
# r = random.randint(0, mnist.test.num_examples - 1)
# print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
# print("Prediction: ", sess.run(
#     tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

'''
Epoch: 0001 cost = 0.340291267
Epoch: 0002 cost = 0.090731326
Epoch: 0003 cost = 0.064477619
Epoch: 0004 cost = 0.050683064
Epoch: 0005 cost = 0.041864835
Epoch: 0006 cost = 0.035760704
Epoch: 0007 cost = 0.030572132
Epoch: 0008 cost = 0.026207981
Epoch: 0009 cost = 0.022622454
Epoch: 0010 cost = 0.019055919
Epoch: 0011 cost = 0.017758641
Epoch: 0012 cost = 0.014156652
Epoch: 0013 cost = 0.012397016
Epoch: 0014 cost = 0.010693789
Epoch: 0015 cost = 0.009469977
Learning Finished!
Accuracy: 0.9885
'''
