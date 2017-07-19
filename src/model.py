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

    def __init__(self, FeaturedData, model = None):
        self.curr_model = None
        self.df_pred = None
        self.model = model
        #Load train_test data.
        self.FeaturedData = FeaturedData

    def _load_data(self):
        '''
        Load X_train and y_train data
        '''
        self.df_X_train = self.FeaturedData.df_X_train
        #Handle response data.
        self.df_y_train = self.FeaturedData.df_y_train

    def _splitter(self):
        '''
        Split and vectorize data for model
        '''
        self.df_y_train.rename(columns = {'reordered' : 'target', 'order_id': 'target_order'}, inplace = True)
        #Index only needed cols.
        self.df_y_train = self.df_y_train.loc[:,['product_id', 'target', 'user_id']].copy()
        #Merge X_train to y_train
        #if product is new as has not been ordered before, then it is not relevant for merge.
        #all products that have a previous order but are not in new one should be zero.
        self.df = self.df_X_train.merge(self.df_y_train, on = ["user_id",'product_id'], how = 'left')
        self.df.loc[self.df.target.isnull(), 'target'] = 0
        self.df.days_since_prior_order.fillna(0, inplace = True)

        #Had to use this hacky way of getting the target col because it was outputing inf values non=sense on array values.
        cols = self.df.columns.tolist()
        cols.append(cols.pop(cols.index('target')))
        #Vectorize X and y
        self.df = self.df.loc[:,cols].copy()

        self.X = self.df.values[:,:-1]
        self.y = self.df.values[:,-1]

        return self.X, self.y

    def _model(self):
        '''
        First iteration will be a random forrest.
        '''
        # Init model
        self.model = RF(n_estimators = 10)
        X, y = self._splitter()
        self.model.fit(X, y)

    def fit(self):
        '''
        Run chosen model
        '''
        print 'Model %s' %self.curr_model

        self._load_data()
        self._model()

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
