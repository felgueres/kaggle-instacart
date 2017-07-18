import pandas as pd
import numpy as np
from model import ReorderModel

class Featurization(object):
    """Feature Engineering Dataset"""

    def __init__(self, X_train = '../data/X_train.pickle', y_train = '../data/y_train.pickle'):
        #Load train_test data.
        self.df_X_train = pd.read_pickle(X_train)
        self.df_y_train = pd.read_pickle(y_train)

    def col2use(self):
        '''
        After featurization methods, index dataframe to only consider
        '''
        cols = ['order_id',
        'product_id',
        'add_to_cart_order',
        'reordered',
        'user_id',
        'order_number',
        'order_dow',
        'order_hour_of_day',
        'days_since_prior_order',
        'aisle_id',
        'department_id']

        self.df_X_train = self.df_X_train.loc[cols].copy()

    def transform(self):
        '''
        Transform the feature space.
        '''
        #Index cols2use only.
        self.col2use()

if __name__ == '__main__':
    c = FeaturedData()
    c.transform()
