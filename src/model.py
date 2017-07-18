import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from score import score

class ReorderModel(object):
    """
    Predict which previously purchased products will be in a user’s next order.
    """

    def __init__(self, X_train = '../data/X_train.pickle', y_train = '../data/y_train.pickle', model = 'greedier'):
        self.df_X_train = pd.read_pickle(X_train)
        self.df_y_train = pd.read_pickle(y_train)
        self.y_train_labels = self.df_y_train.groupby(['user_id', 'order_id'])['product_id'].apply(get_products).reset_index()
        self.y_train_labels.rename(columns = {'product_id': 'y'}, inplace = True)
        self.curr_model = None
        self.df_pred = None
        self.model = model

    def greedy(self):
        '''
        This model is the utter baseline - says users will rebuy whatever they bought previously.
        '''
        self.curr_model = 'Greedy-Baseline'
        self.df_pred = self.df_X_train.groupby(self.df_X_train.user_id)['product_id'].apply(get_products).reset_index()
        self.df_pred.rename(columns = {'product_id': 'y_pred'}, inplace = True)

    def greedier(self):
        '''
        This model is a bit more intelligent, it will only take into account items that have been reordered in the past.
        '''
        self.curr_model = 'Greedier'
        self.df_pred = self.df_X_train.loc[self.df_X_train.reordered == 1].groupby('user_id')['product_id'].apply(get_products).reset_index()
        self.df_pred.rename(columns = {'product_id': 'y_pred'}, inplace = True)

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
        # PRINT THE SCORE
        # print "Average F1 Score: {0: .1f}%".format(self.results.f1.mean())

    def fit_predict(self):
        '''
        Run chosen model
        '''
        #Run specified model
        if self.model == 'greedy':
            self.greedy()

        elif self.model == 'greedier':
            self.greedier()

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
    products = [str(product) for product in set(user_products)]
    #concatenate products
    concat_str = ' '.join(products)
    return concat_str

if __name__ == '__main__':
    b = ReorderModel()
    b.fit_predict()
