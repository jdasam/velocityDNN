import numpy as np
import h5py
import scipy.io as sio
from scipy.stats import norm
import tensorflow as tf
import random

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
    whole_x, whole_y = loadTrainSet(whole_file)
    print whole_x[1,2,3]
    n_data = whole_x.shape[0]
    print whole_x.shape, whole_y.shape
    idx = range(n_data)
    random.shuffle(idx)
    n_train = int(n_data*0.6)
    n_valid = int(n_data*0.2)

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

    for setIndex in range(vel_cel.shape[1]):
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
                    tempX = spec_array.reshape((445, 14))
                    dataSetX.append(tempX/np.amax(tempX))
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
    return dataSetX, dataSetY


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
        tempX = np.asarray(test_contents['onsetClusterArray'][0][i]).reshape((445, 14,1))
        pieceX.append(tempX / np.amax(tempX))
        # pieceY.append(np.asarray(test_contents['onsetMatchedVel'][0][i]).reshape(1))
        pieceY.append(velocityToOneHot(test_contents['onsetMatchedVel'][0][i]).reshape(13))

    pieceX = np.asarray(pieceX)
    pieceY = np.asarray(pieceY)
    return pieceX, pieceY


def velocityToOneHot(velocity):

    oneHotVec = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])
    threshold = Threshold()
    # print(threshold.binaryIndexOf(int(94.0)))

    # index, remainder = threshold.binaryIndexOf(velocity)

    index = threshold.binaryIndexOf(velocity)

    diffA = velocity - threshold.value[index]
    diffB = threshold.value[index+1] - velocity
    remainder = diffA / float(diffA + diffB)
    # print(velocity)
    # print('binaryIndexof Result: ', index)

    oneHotVec[index] = 1
    # if index > 0:
    #     oneHotVec[index] = 1 - abs(0.5 - remainder)
    #     oneHotVec[index-1] = 0.5 - max(0, 0.5-remainder)
    # else:
    #     oneHotVec[index] = 1
    # if index < len(oneHotVec) -1:
    #     oneHotVec[index+1] = max(0.5, remainder) - 0.5
    return oneHotVec

def iteratorToOneHot(iterator):
    print (iterator.shape)
    for i in range(iterator.shape[0]):
        iterator[i] = velocityToOneHot(iterator[i])
    return iterator



class Threshold(list):
    def __init__(self):
        list.__init__([])
        self.value = [0, 20, 30, 40, 50, 55, 60, 65, 70, 80, 90, 100, 110, 127]

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

def resultToStatistic(resultVector):
    velocityVector = [None] * resultVector.shape[0]
    for i in range(resultVector.shape[0]):
        velocityVector[i] = oneHotToVelocity(resultVector[i, :])

    print(velocityVector)
    (mu, sigma) = norm.fit(velocityVector)
    return mu, sigma




if __name__ == '__main__':
    convert_to('R8dataS2Gpre20Ubn15UibId100Hb15postItr30_20_1_50_1_1000.mat', 'dataInterpol')