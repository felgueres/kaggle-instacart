import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier as RF
from score import score
from featurization import FeaturedData
from utilities_funcs import get_products

class ReorderModel(object):
    """
    Predict which previously purchased products will be in a user next order.
    """

    def __init__(self, FeaturedData = None, isbaseline = False, model = None):
        self.curr_model = None
        self.df_pred = None
        self.model = model
        self.isbaseline = isbaseline
        #Load train_test data.
        self.FeaturedData = FeaturedData

    def _load_data(self):
        '''
        Load X_train and y_train data
        '''
        if self.isbaseline:
            self.df_X_train = pd.read_pickle('../data/X_train.pickle')
            self.df_y_train = pd.read_pickle('../data/y_train.pickle')
            self.y_train_labels.loc[self.df_y_train.reordered == 0, 'product_id'] = 'None'
            self.y_train_labels = self.df_y_train.groupby(['user_id', 'order_id'])['product_id'].apply(get_products).reset_index()
            self.y_train_labels.rename(columns = {'product_id': 'y'}, inplace = True)

        elif FeaturedData:

            self.df_X_train = self.FeaturedData.df_X_train
            #Handle response data.
            self.df_y_train = self.FeaturedData.df_y_train

        else:
            return 'No data.'

    def _model(self):
        '''
        First iteration will be a random forrest.
        '''
        # Init model
        self.model = RF(n_estimators = 50)
        X, y = self._splitter()
        self.model(X, y)

    def _splitter(self):
        '''
        Split and vectorize data for model
        '''
        self.df_y_train.rename(columns = {'reordered' : 'y'})
        #Index only needed cols.
        self.df_y_train = self.df_y_train.loc[:,['order_id','product_id','y_pred']].copy()
        #Merge X_train to y_train
        self.df = self.df_y_train.merge(self.df_X_train, on = ["order_id",'product_id'], how = 'left')
        #Vectorize X and y
        # return X, y

    def greedy(self):
        '''
        This model is the utter baseline - says users will rebuy whatever they bought previously.
        '''
        self.curr_model = 'Greedy'
        self.df_pred = self.df_X_train.groupby('user_id')['product_id'].apply(get_products).reset_index()
        self.df_pred.rename(columns = {'product_id': 'y_pred'}, inplace = True)

    def greedier(self):
        '''
        This model is a bit more intelligent, it will only take into account items that have been reordered in the past.
        '''
        self.curr_model = 'Greedier'
        self.df_pred = self.df_X_train
        self.df_pred.loc[self.df_pred.reordered == 0, 'product_id'] = 'None'
        self.df_pred = self.df_pred.groupby('user_id')['product_id'].apply(get_products).reset_index()
        self.df_pred.rename(columns = {'product_id': 'y_pred'}, inplace = True)

    def baseline(self):
        '''
        Take into account items that have been reordered in the last order.
        '''
        self.curr_model = 'Baseline'
        self.df_pred = self.df_X_train
        self.df_pred.loc[self.df_pred.reordered == 0, 'product_id'] = 'None'
        mask = self.df_pred.groupby('user_id')['order_number'].transform(max)
        self.df_pred = self.df_pred[self.df_pred.order_number == mask].groupby('user_id')['product_id'].apply(get_products).reset_index()
        self.df_pred.rename(columns = {'product_id': 'y_pred'}, inplace = True)

    def fit(self):
        '''
        Run chosen model
        '''
        #Run specified model
        if self.model == 'greedy':
            self.greedy() #Yield f1Score 21.528%

        elif self.model == 'greedier':
            self.greedier() #Yields f1Score: 30.267%

        elif self.model == 'baseline':
            self.baseline() #Yields f1 Score: 32.575%

        print 'Model %s' %self.curr_model

        self._load_data()
        self._splitter()


    def scoremodel(self):
        '''
        Compute f1score for the model
        '''
        #Merge actual vs predictions
        self.df_y_vs_y_pred = self.y_train_labels.merge(self.df_pred, on = "user_id", how = 'left')
        #Get actual vs pred columns in tuples in iterator.
        self.df_y_vs_y_pred = self.df_y_vs_y_pred.loc[:, ['y', 'y_pred']].copy()
        #Compute results: Results columns at this point are: index, y, y_pred
        self.results = [score(order[1], order[2]) for order in self.df_y_vs_y_pred.itertuples()]
        #Insert results to dataframe
        self.results = pd.DataFrame(np.array(self.results), columns = ['precision', 'recall', 'f1'])
        # PRINT SCORE
        print "Average F1 Score: {0:.3%}".format(self.results.f1.mean())


if __name__ == '__main__':
    data = FeaturedData()
    data.transform()

    model = ReorderModel(FeaturedData = data)
    model.fit()
