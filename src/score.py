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

def get_them(user):
    '''
    Get list of product ids from a groupby object into competitions format
    #Sample:
    #order_id,products
    #17,1 2
    #34,None
    #137,1 2 3
    '''
    #Get list of product ids from a grouped object

    products = [str(product) if reorder == 1 else '' for reorder in user['reordered'] for product in set(user['product_id'])]

    #concatenate products
    concat_str = ' '.join(products)

    return concat_str





if __name__ == '__main__':
    pass
