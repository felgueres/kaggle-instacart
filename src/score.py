import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

def score(y, y_pred):
    '''
    Compute score metrics of classification -- Precision, Recall and F1

    Parameters
    ----------
    y: Numpy array
        Array of actual products

    y_pred: Numpy array
        Array with model predictions

    Output
    ------
    Predicion, recall, f1
    '''

    print y.__class__

    y = y.split(' ')

    y_pred = y_pred.split(' ')

    rr = (np.intersect1d(y, y_pred))

    precision = np.float(len(rr)) / len(y_pred)

    recall = np.float(len(rr)) / len(y)

    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)

if __name__ == '__main__':
    pass
