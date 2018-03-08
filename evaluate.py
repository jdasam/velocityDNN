import numpy as np

def myAccu(y,pred):
    # y and pred should be np array with shape of (time, feature)
    yy = y.flatten()
    ppred = pred.flatten()

    TP = np.sum(((2*yy - ppred) == 1).astype(np.int))
    FN = np.sum(((yy - ppred) == 1).astype(np.int))
    FP = np.sum(((yy - ppred) == -1).astype(np.int))

    '''
    print('diff,GR,P',str(temp),str(temp_yy),str(temp_ppred))
    print('TP,FN,FP',TP,FN,FP)
    print('totalLen',yy.shape, ppred.shape, len(yy))
    '''

    precision =TP/float(TP+FP)
    recall = TP/float(TP+FN)
    # accuracy = TP/float(TP+FN+FP)

    F = 2*precision*recall/float(precision+recall)

    totalAccuracy = (len(yy) - np.sum(np.abs(yy - ppred))) * 100.0 / len(yy)
    '''
    print('precision, recall, accuracy, F1-score',precision, recall, accuracy,F)
    print('totalAccuracy',str(totalAccuracy))

    return yy, ppred, accuracy
    '''
    return precision, recall, totalAccuracy, F


def find_thresh(y, pred, threshold=0.2, wait=20, verbose=1):
    best_thresh = 0
    f = 0
    wait_count = 0
    while True:
        thresh_pred = (pred > threshold).astype(np.int)
        _, _, _, temp_f = myAccu(y, thresh_pred)
        if verbose == 1:
            print('threshold: %2f, F1-score=%5f' % (threshold, temp_f))
        if temp_f > f:
            if verbose == 2:
                print('threshold: %2f, F1-score=%5f' % (threshold, temp_f))
            f = temp_f
            best_thresh = threshold
            wait_count = 0
        else:
            wait_count += 1

        threshold += 0.01
        if wait_count >= wait or threshold > 0.7:
            break

    return best_thresh


def evaluate(predict, y_test):
    if predict.ndim == 3:
        predict = np.reshape(predict, [predict.shape[0] * predict.shape[1], predict.shape[2]])
    if y_test.ndim == 3:
        y_test = np.reshape(y_test, [y_test.shape[0] * y_test.shape[1], y_test.shape[2]])

    threshold = find_thresh(y_test, predict, verbose=0)

    best_pred = (predict > threshold).astype(np.int)
    precision1, recall1, accuracy1, F1 = myAccu(y_test, best_pred)

    fixed_pred = (predict > 0.5).astype(np.int)
    precision2, recall2, accuracy2, F2 = myAccu(y_test, fixed_pred)

    return threshold, {'precision': precision1, 'recall': recall1, 'accuracy': accuracy1, 'F': F1}, \
           {'precision': precision2, 'recall': recall2, 'accuracy': accuracy2, 'F': F2}
