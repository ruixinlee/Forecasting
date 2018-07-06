from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import pandas as pd


selected_col = ['IHS_t', 'market','target_month','selling_days','vehicle_line']

class variables_select_Transformer(BaseEstimator,TransformerMixin):
    def __init__(self, cols=selected_col):
        self.cols = cols

    def fit(self, X, y=None):
        X = X[self.cols]
        self.input_shape_ = X.shape
        return(self)

    def transform(self, X):
        X = X[self.cols]
        return(X)



class sarimax(BaseEstimator,RegressorMixin):
    def __init__(self, granularity = None, params = None):
        self.params = params
        self.granularity = granularity
        self.selling_days = 'selling_days'

    def transform_v_y(self,v, y = None):
        v = v.reset_index().set_index('target_month')
        if y is not None:
            y = y.loc[v['index']]
            y.index = v.index
            y_v = y/ v[self.selling_days]
        else:
            y_v = None
        return(v, y_v)

    def inv_trasform_y(self, v, y_v):
        if y_v is not None:
            y_v = y_v.loc[v['index']]
            y = y_v*v[self.selling_days]
            y.index = v['index']

        v = v.reset_index().set_index('index')
        return(v, y)

    def fit(self, X,y):
        X_dict=dict(tuple(X.groupby(self.granularity))) #create a dictionary of dataframes according to the granularity of model. Here, for each market and vehicle line we have a df
        self.model = {}
        self.fitted_model = {}
        for k,v in X_dict.items():
            v, y_v = self.transform_v_y(v,y)
            self.model[k] = sm.tsa.statespace.SARIMAX(y_v.tolist(), v['IHS_t'].tolist())
            self.fitted_model[k] = self.model[k].fit()
        return(self)

    def predict(self, X):
        X_dict = dict(tuple(X.groupby(self.granularity)))
        pred = []
        for k, v in X_dict.items():
            v,_ = self.transform_v_y(v)
            pred = self.fitted_model[k].get_prediction(exog=v['IHS_t']).predicted_mean
        predictions = pd.concat(pred)

        ## need to think about index
        return(predictions)


def construct_Pipline():
    var_trans = variables_select_Transformer()
    pline = Pipeline([('var_trans', var_trans)
                                ,('sarimax', sarimax(granularity= ['market', 'vehicle_line']))])

    return(pline)

