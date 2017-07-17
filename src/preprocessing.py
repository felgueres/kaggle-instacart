import pandas as pd
import numpy as np
from os import path, listdir

class Preprocess(object):

    '''
    Handles preprocessing of data for Instacart Competition.

    Parameters
    ----------
    path: str
        Path to folder with data files.
    '''

    def __init__(self, datapath = '../data/'):
        '''
        Initialize with string to data folder.
        '''
        self.files = [path.join(datapath, file) for file in listdir(datapath) if file.endswith('csv')]

    def _load2df(self):
        '''
        Load data into dataframes
        '''
        # Load data to dataframe
        self._df_aisles = pd.read_csv(self.files[0])
        self._df_dpts = pd.read_csv(self.files[1])
        self.df_order_prior = pd.read_csv(self.files[2])
        self.df_order_train = pd.read_csv(self.files[3])
        self.df_orders = pd.read_csv(self.files[4])
        self.df_products = pd.read_csv(self.files[5])

    def _products(self):
        '''
        Merge product-related dataframes.
        '''
        #Merge aisles, dpt and products -- product is left
        self.df_products = self.df_products.merge(self._df_aisles, left_on= 'aisle_id', right_on= 'aisle_id')
        self.df_products = self.df_products.merge(self._df_dpts, left_on = 'department_id', right_on= 'department_id')

    def _users(self):
        '''
        Get a list of users corresponding to the training and testing dataset.
        '''
        users_train_all = self.df_orders.loc[(self.df_orders.eval_set == "train")].user_id
        #The users test dataset is the hold out only for final submissions
        self.users_test = self.df_orders.loc[(self.df_orders.eval_set == "test")].user_id
        #Divide the train dataset into a training and validation dataset
        #training dataset
        self.users_train = users_train_all.sample(frac = 0.8, random_state = 10)
        #validation dataset / users that are not present in the users train set
        self.users_val = users_train_all[~users_train_all.isin(self.users_train)]

    def _partition(self):
        '''
        Separate the training datapoints into validation and test set.
        '''
        pass

    def fit(self):
        '''
        Runs all preprocessing methods. Output to be used by the model class.
        '''
        self._load2df()
        self._users()
        self._products()
        self._partition()


class FeaturedData(object):
    """Feature Engineering Dataset"""

    def __init__(self, Preprocess):
        self.data = Preprocess


    def _somemethodstofeaturize(self):
        '''
        FeatureSpace to learn from.
        '''
        pass

if __name__ == '__main__':
    a = Preprocess()
    a.fit()
