from __future__ import division
import math
import numpy as np
import h5py
import scipy.io as sio
from scipy.stats import norm
import tensorflow as tf
import random
import os

velClassNum = 8
specLength = 14

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_to(whole_file, save_folder):

    def writer(set_name):
        temp_writer = tf.python_io.TFRecordWriter(save_folder + '/' + '{}_.tfrecords'.
                                         format(set_name))
        return temp_writer

    t_writer = writer('train')
    v_writer = writer('valid')
    te_writer = writer('test')
    # whole_x, whole_y = loadTrainSet(whole_file)
    whole_x, whole_y = mergeMatFolder(whole_file)
    print whole_x[1,2,3]
    n_data = whole_x.shape[0]
    print whole_x.shape, whole_y.shape
    idx = range(n_data)
    random.shuffle(idx)
    n_train = int(n_data*0.7)
    n_valid = int(n_data*0.3)

    for n in range(n_data):
        if n % 5000 == 0:
            print n
        data_x = whole_x[idx[n], :, :]
        data_y = whole_y[idx[n],:]
        example = tf.train.Example(features=tf.train.Features(feature={
            'feature': _float_feature(data_x.flatten()),
            'label': _float_feature(data_y.flatten())}))
        if n <= n_train:
            t_writer.write(example.SerializeToString())
        elif n <= n_train + n_valid:
            v_writer.write(example.SerializeToString())
        else:
            te_writer.write(example.SerializeToString())
    t_writer.close()
    v_writer.close()
    te_writer.close()


def loadTrainSet(filename):
    mat_contents = h5py.File(filename, 'r')
    vel_cel = mat_contents['velocityGainMatchingCell']

    dataSetX = []
    dataSetY = []

    for setIndex in range(3): #range(vel_cel.shape[1]):
        for foldIndex in range(vel_cel.shape[0]):
            set_fold_cell = mat_contents[vel_cel[foldIndex, setIndex]]
            spec_cell = mat_contents[set_fold_cell[3, 0]]
            gt_vel_cell = mat_contents[set_fold_cell[2,0]]
            print(set_fold_cell.shape)
            for j in range(spec_cell.shape[0]):
                print('Set:', setIndex, 'fold:',foldIndex,'pitch:',j ,'dataSize:',len(dataSetY))
                for dataSize in range (spec_cell.shape[1]):
                    spec_array = np.transpose(np.asarray(mat_contents[spec_cell[j, dataSize]]))
                    if not spec_array.any():
                        break
                    tempX = spec_array.reshape((445, specLength))
                    dataSetX.append(tempX/np.amax(tempX))
                    # dataSetX.append(tempX)
                    # dataSetY.append(np.asarray(vel_cel[setIndex, foldIndex][0, 2][i, j].reshape(1)))
                    # print('gt_Vel:' ,np.asarray(gt_vel_cell[j,dataSize]))


                    # dataSetY.append(np.asarray(gt_vel_cell[j,dataSize]).reshape(1))
                    dataSetY.append(velocityToOneHot(gt_vel_cell[j,dataSize]).reshape(velClassNum))


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
    return dataSetX, dataSetY

def mergeMatFolder(path):
    fileList = os.listdir(path)
    wholeFileX = np.empty([0, 445,specLength ,1])
    wholeFileY = np.empty([0, velClassNum])
    for matIndex in range(len(fileList)):
        pieceX, pieceY = loadPiece(path+'/'+fileList[matIndex])
        wholeFileX = np.concatenate((wholeFileX,pieceX), axis=0)
        wholeFileY = np.concatenate((wholeFileY,pieceY), axis=0)
    return wholeFileX, wholeFileY


def dataSetToTrainTest(dataSetX, dataSetY, trainSetSize):

    trainSetX = dataSetX[0:trainSetSize,:,:]
    trainSetY = dataSetY[0:trainSetSize,:]
    testSetX = dataSetX[trainSetSize:len(dataSetX),:,:]
    testSetY = dataSetY[trainSetSize:len(dataSetY),:]


    return trainSetX, trainSetY, testSetX, testSetY



def unison_shuffled_copies(a,b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def loadPiece(fileName):
    test_contents = sio.loadmat(fileName)
    pieceSetSize = len(test_contents['onsetClusterArray'][0])

    pieceX = []
    pieceY = []
    for i in range(pieceSetSize):
        # print(test_contents['onsetClusterArray'][0][i])
        tempX = np.asarray(test_contents['onsetClusterArray'][0][i]).reshape((445, specLength,1))
        pieceX.append(tempX / np.amax(tempX))
        # pieceY.append(np.asarray(test_contents['onsetMatchedVel'][0][i]).reshape(1))
        pieceY.append(velocityToOneHot(test_contents['onsetMatchedVel'][0][i]).reshape(velClassNum))

    pieceX = np.asarray(pieceX)
    pieceY = np.asarray(pieceY)
    return pieceX, pieceY


def velocityToOneHot(velocity):

    # oneHotVec = np.array([0,0,0,0,0,0,0,0])
    oneHotVec = np.zeros(velClassNum, dtype=float)
    threshold = Threshold()
    # print(threshold.binaryIndexOf(int(94.0)))

    # index, remainder = threshold.binaryIndexOf(velocity)

    index = threshold.binaryIndexOf(velocity)

    diffA = velocity - threshold.value[index]
    diffB = threshold.value[index+1] - velocity
    remainder = diffA / float(diffA + diffB)
    # print(velocity)
    # print('binaryIndexof Result: ', index)

    # oneHotVec[index] = 1
    oneHotVec[index] = 1 - abs(0.5 - remainder)
    if index > 0:
        oneHotVec[index-1] = max(0, 0.5 - remainder)
    if index < len(oneHotVec) -1:
        oneHotVec[index+1] = max(0.5, remainder) - 0.5
    return oneHotVec

def iteratorToOneHot(iterator):
    print (iterator.shape)
    for i in range(iterator.shape[0]):
        iterator[i] = velocityToOneHot(iterator[i])
    return iterator



class Threshold(list):
    def __init__(self):
        list.__init__([])
        self.value = [0, 30, 40, 50, 60, 70, 80, 95, 127]

    def binaryIndexOf(self, searchElement):
        minIndex = 0
        maxIndex = len(self.value) -1

        if searchElement < self.value[minIndex]:
            return 0

        while minIndex < maxIndex:
            currentIndex = (minIndex + maxIndex) //2
            currentElement = self.value[currentIndex]

            if currentElement < searchElement:
                if self.value[currentIndex+1]>searchElement:
                    return currentIndex
                else: minIndex = currentIndex +1
                if minIndex == maxIndex & self.value[maxIndex] > searchElement:
                    return currentIndex
            else:
                maxIndex = currentIndex -1

        intIndex = min(minIndex,maxIndex)
        # diffA = searchElement - self.value[intIndex]
        # diffB = self.value[intIndex+1] - searchElement

        # remainder = diffA / float(diffA + diffB)
        return intIndex


def oneHotToVelocity(vector):
    threshold = Threshold()

    index = np.argmax(vector)

    velocity = int((threshold.value[index + 1] - threshold.value[index]) / 2 +  threshold.value[index])
    return velocity

def velocityToStatistic(velocityVector):
    (mu, sigma) = norm.fit(velocityVector)
    return mu, sigma

def resultToVelocity(resultVector):
    velocityVector = [None] * resultVector.shape[0]
    for i in range(resultVector.shape[0]):
        velocityVector[i] = oneHotToVelocity(resultVector[i, :])

    return velocityVector

def runTest(sess, file_name, batch_size):
    pieceX, pieceY = loadPiece(file_name)
    print(pieceX)
    test_batch = int(math.ceil(pieceX.shape[0]/batch_size))
    print('number of test batch: ', test_batch)
    result = np.empty([1,velClassNum])
    avg_pieceAccu = 0
    for i in range(test_batch):
        batch_xs, batch_ys = pieceX[i * batch_size:(i + 1) * batch_size], pieceY[
                                                                             i * batch_size:(i + 1) * batch_size]
        batch_xs = batch_xs.reshape([-1, 6230])
        feed_dict = {X:batch_xs, Y: batch_ys, is_training: False}
# c, validAccu, validError = sess.run([cost, accuracy, mean_error], feed_dict=feed_dict)
        pieceAccu, tempResult = sess.run([accuracy, hypothesis], feed_dict=feed_dict)
        avg_pieceAccu += pieceAccu
        print('TempResult: ', tempResult.shape)
        result = np.concatenate((result, tempResult), axis=0)
        # result = result.flatten()
    pieceAccu = avg_pieceAccu / test_batch
    # pieceAccu,result =  sess.run([(accuracy), hypothesis], feed_dict={X: pieceX, Y: pieceY, is_training:False})
    # result =  sess.run(hypothesis, feed_dict={X: pieceX, Y: pieceY, is_training:False})
    return result

def readExtInFolder(dir, ext):
    fileList = os.listdir(dir)
    extList = []
    for f in fileList:
        if f.endswith('.'+  ext):
            extList.append(f)

    extList.sort()
    return extList

def calError(label, hypothesis):
    label_vel = np.array(resultToVelocity(label))
    result_vel = np.array(resultToVelocity(hypothesis))

    error = np.abs(label_vel-result_vel)
    mean_error = np.mean(error)

    return mean_error





if __name__ == '__main__':
    convert_to('./mat4window', 'dataWindow4')