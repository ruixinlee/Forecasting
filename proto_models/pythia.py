"""
Pythia - The Prophet Channel
In honour of the high priestess of the Temple of Apollo at Delphi
"""

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from fbprophet import Prophet
import pandas as pd
import numpy as np

class Data_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, params=None):
        self.params = params

    def cal_theta(self, X, y):
        X['year'] = X['target_month'].dt.year
        X['quarter'] = X['target_month'].apply(lambda x: (x.month-1)//3 + 1)
        X['month_in_quarter'] = X['target_month'].dt.month - ((X['target_month'].dt.month-1)//3)*3

        X['actual_sales'] = y
        actual_sales_qtr = pd.DataFrame(X.groupby(['market', 'vehicle_line', 'year', 'quarter'])['actual_sales'].apply(sum))
        actual_sales_qtr.rename(columns={'actual_sales': 'actual_sales_qtr'}, inplace=True)
        X = X.join(actual_sales_qtr, on=['market', 'vehicle_line', 'year', 'quarter'])

        return(X)

    def fit(self, X, y=None):

        X =self.cal_theta(X,y)
        X['theta'] = X['actual_sales'] / X['actual_sales_qtr']
        self.thetas = dict(tuple(X[['market', 'vehicle_line', 'quarter', 'month_in_quarter', 'theta']].groupby(['market', 'vehicle_line', 'quarter', 'month_in_quarter'])['theta']))

        return self

    def transform(self, X):
        X['year'] = X['target_month'].dt.year
        X['quarter'] = X['target_month'].apply(lambda x: (x.month-1)//3 + 1)
        X['month_in_quarter'] = X['target_month'].dt.month - ((X['target_month'].dt.month-1)//3)*3

        if 'actual_sales' not in X.columns:
            X['theta'] = np.nan
            for i in range(0, len(X)):
                market = X.iloc[i]['market']
                vehicle_line = X.iloc[i]['vehicle_line']
                quarter = X.iloc[i]['quarter']
                month_in_quarter = X.iloc[i]['month_in_quarter']
                theta = np.mean(self.thetas[market, vehicle_line, quarter, month_in_quarter])
                X.iloc[i, -1] = theta
                self.thetas[market, vehicle_line, quarter, month_in_quarter].append(pd.Series(theta))
        else:
            X = self.cal_theta(X,y)

        X['IHS_t_vl_mth'] = X['IHS_t_vl'] * X['theta']
        X['IHS_t_mth'] = X['IHS_t'] * X['theta']
        return X

class Pythia(BaseEstimator, RegressorMixin):
    def __init__(self, granularity=None, params=None):
        self.params = params
        self.granularity = granularity

    def fit(self, X, y):
        self.predictors_num = 0
        X['y'] = y
        X_dict = dict(tuple(X.groupby(self.granularity))) #create a dictionary of dataframes according to the granularity of model. Here, for each market and vehicle line we have a df
        self.model = {}
        for k,v in X_dict.items():
            self.model[k] = Prophet()
            ds = v['target_month']
            target = v['y']
            prophet_df = pd.DataFrame({'ds': ds, 'y': target})
            self.model[k].fit(prophet_df) #create a model for each k (market and vehicle line)
        return self

    def predict(self, X):
        X_dict = dict(tuple(X.groupby(self.granularity)))
        predictions = []
        for k, v in X_dict.items():
            ds = v['target_month']
            prophet_df = pd.DataFrame({'ds': ds})
            pred = self.model[k].predict(prophet_df)['yhat'] #predict for each market vehicle line using the appropriate model trained in 'fitz
            pred.index = v.index
            predictions.append(pred)
        return pd.concat(predictions)

def construct_Pipline():
    return Pipeline([('pythia_transformer', Data_Transformer()),
                    ('pythia_model', Pythia(granularity=['market', 'vehicle_line']))])
