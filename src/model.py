import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from score import score

class ReorderModel(object):
    """
    Predict which previously purchased products will be in a user’s next order.
    """

    def __init__(self, X_train, istraining = False, model = 'greedier'):
        self.curr_model = None
        self.df_pred = None
        self.model = model
        #Load train_test data.
        if istraining:
            self.df_X_train = pd.read_pickle('../data/X_train.pickle')

        self.df_X_train = X_train

        self.df_y_train = pd.read_pickle('../data/y_train.pickle')
        '''
        Handle the response dataset
        '''
        #Filter for only reordered items
        self.y_train_labels = self.df_y_train
        self.y_train_labels.loc[self.df_y_train.reordered == 0, 'product_id'] = 'None'
        self.y_train_labels = self.df_y_train.groupby(['user_id', 'order_id'])['product_id'].apply(get_products).reset_index()
        self.y_train_labels.rename(columns = {'product_id': 'y'}, inplace = True)

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

    def scoremodel(self):
        '''
        Compute f1score for the model
        '''
        #Merge actual vs predictions
        self.df_y_vs_y_pred = self.y_train_labels.merge(self.df_pred, on = "user_id", how = 'left')
        #Drop nulls
        self.df_y_vs_y_pred.dropna(axis = 0, inplace = True)
        #Get actual vs pred columns in tuples in iterator.
        self.df_y_vs_y_pred = self.df_y_vs_y_pred.loc[:, ['y', 'y_pred']].copy()
        #Compute results: Results columns at this point are: index, y, y_pred
        self.results = [score(order[1], order[2]) for order in self.df_y_vs_y_pred.itertuples()]
        #Insert results to dataframe
        self.results = pd.DataFrame(np.array(self.results), columns = ['precision', 'recall', 'f1'])
        # PRINT SCORE
        print "Average F1 Score: {0:.3%}".format(self.results.f1.mean())

    def fit_predict(self):
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
        self.scoremodel()

def get_products(user_products):
    '''
    Get list of product ids from a groupby object into competitions format
    #Sample:
    #order_id,products
    #17,1 2
    #34,None
    #137,1 2 3
    '''
    #Get list of product ids from a grouped object
    products = [str(product) for product in set(user_products) if not product == 'None']
    #Check if list is empty, which would mean there are no reordered items; if so, replace by a single None.
    if not products:
        products.append('None')
    #concatenate products
    concat_str = ' '.join(products)
    return concat_str

if __name__ == '__main__':
    b = ReorderModel()
    b.fit_predict()
