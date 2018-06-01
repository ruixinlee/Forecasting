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

    def fit(self, X, y=None):
        X['target_year'] = X['target_month'].dt.year
        X['target_qtr'] = X['target_month'].apply(lambda x: (x.month-1)//3 + 1)
        X['target_month'] = X['target_month'].dt.month - ((df['target_month'].dt.month-1)//3)*3

        actual_sales_qtr = pd.DataFrame(X.groupby(['market', 'vehicle_line', 'target_year', 'target_qtr'])['actual_sales'].apply(array).apply(sum))
        actual_sales_qtr.rename(columns={'actual_sales': 'actual_sales_qtr'}, inplace=True)
        X = X.join(actual_sales_qtr, on=['market', 'vehicle_line', 'target_year', 'target_qtr'])

        X['theta'] = X['actual_sales'] / X['actual_sales_qtr']
        self.thetas = dict(tuple(X[['market', 'vehicle_line', 'target_qtr', 'target_month', 'theta']].groupby(['market', 'vehicle_line', 'target_qtr', 'target_month'])['theta']))

        X['IHS_t_vl_mth'] = X['IHS_t_vl'] * X['theta']
        X['IHS_t_mth'] = X['IHS_t'] * X['theta']
        return self

    def transform(self, X):
        X['target_year'] = X['target_month'].dt.year
        X['target_qtr'] = X['target_month'].apply(lambda x: (x.month-1)//3 + 1)
        X['target_month'] = X['target_month'].dt.month - ((df['target_month'].dt.month-1)//3)*3

        X['theta'] = np.nan
        for i in range(0, len(X)):
            market = X.iloc[i]['market']
            vehicle_line = X.iloc[i]['vehicle_line']
            qtr = X.iloc[i]['target_qtr']
            month = X.iloc[i]['target_month']
            theta = np.mean(self.thetas[market, vehicle_line, qtr, month])
            X.iloc[i, -1] = theta
            self.thetas[market, vehicle_line, qtr, month].append(pd.Series(theta))

        X['IHS_t_vl_mth'] = X['IHS_t_vl'] * X['theta']
        X['IHS_t_mth'] = X['IHS_t'] * X['theta']
        return X

class Pythia(BaseEstimator, RegressorMixin):
    def __init__(self, granularity=None, params=None):
        self.params = params
        self.granularity = granularity

def create_ihs_month_forecast(df):
    df['IHS_t_vl_mth'] = df['IHS_t_vl'] * df['theta']
    df['IHS_t_mth'] = df['IHS_t'] * df['theta']
    return df

    def fit(self, X, y):
        X_dict = dict(tuple(X.groupby(self.granularity))) #create a dictionary of dataframes according to the granularity of model. Here, for each market and vehicle line we have a df
        self.model = []
        for k,v in X_dict.items():
            self.model[k] = Prophet()
            ds = v['target_month']
            target = np.log(y)
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
        predictions = pd.concat(pred)
        return predictions

def construct_Pipline():
    data_transformer = Data_Transformer()
    model = Pythia(granularity=['market', 'vehicle_line'])
    pline = Pipeline([('data_transformer', data_transformer),
                    ('model', model)])
    return pline
