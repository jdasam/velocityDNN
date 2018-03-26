from sklearn import svm
import loadMatToNumpy as loadMat
import numpy as np


# pieceX, pieceY = loadMat.loadPiece('./matSMDtrain/Bach_BWV849-01_001_20090916-SMD.mp3.mat')
trainX, trainY = loadMat.mergeMatFolder('./matSMDtrain')
trainX = trainX.reshape([trainX.shape[0], -1])
trainY = np.argmax(trainY, axis=1)

# testX, testY = loadMat.mergeMatFolder('./matSMDtest')
# testX = testX.reshape([testX.shape[0], -1])
# testY = np.argmax(testY, axis=1)


print(trainY.shape)

clf = svm.SVC(decision_function_shape='ovo')
clf.fit(trainX, trainY)

print(clf)

# clf.predict(testX)
