import numpy as np
import h5py


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
                    tempX = spec_array.reshape((445, 14, 1))
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
    threshold = Threshold()
    # print(threshold.binaryIndexOf(int(94.0)))

    # index, remainder = threshold.binaryIndexOf(velocity)

    index = threshold.binaryIndexOf(velocity)

    diffA = velocity - threshold.value[index]
    diffB = threshold.value[index+1] - velocity
    remainder = diffA / float(diffA + diffB)
    # print(velocity)
    # print('binaryIndexof Result: ', index)


    if index > 0:
        oneHotVec[index] = 1 - abs(0.5 - remainder)
        oneHotVec[index-1] = 0.5 - max(0, 0.5-remainder)
    else:
        oneHotVec[index] = 1
    if index < len(oneHotVec) -1:
        oneHotVec[index+1] = max(0.5, remainder) - 0.5
    return oneHotVec

def binaryIndexOf(searchElement):
    minIndex = 0
    maxIndex = len(self)

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

