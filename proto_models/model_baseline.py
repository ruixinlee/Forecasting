from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline


class X_ColSelect_Transformer(BaseEstimator,TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols

    def fit(self, X, y=None):
        X = X[self.cols]
        self.input_shape_ = X.shape
        return(self)

    def transform(self, X):
        X = X[self.cols]
        return(X)

class X_Date_Transformer(BaseEstimator,TransformerMixin):
    def __init__(self, params = None):
        self.params = params

    def fit(self, X, y = None):
        return(self)

    def transform(self, X):
        X['month'] = X.target_month.dt.month
        X['year'] = X.target_month.dt.year
        return(X)

class Baseline_Model(BaseEstimator,RegressorMixin):
    def __init__(self, params = None):
        self.params = params

    def fit(self, X,y):
        self.predictors_num = 0

        self.variables = ['vehicle_line', 'market', 'month']
        idx = X.groupby(self.variables).agg({'target_month': ['idxmax']})
        idx = idx.iloc[:, 0].tolist()

        df_pred = X.loc[idx]
        df_pred = df_pred[self.variables]
        df_target = y.loc[idx]
        df_pred = df_pred.merge(df_target.to_frame(), left_index=True, right_index=True, how='inner')
        self.model = df_pred

        return(self)

    def predict(self, X):
        #predictions should be a panda series with the original row index
        X = X.reset_index().merge(self.model, on=self.variables, how='inner').set_index('index')
        predictions = X['actual_sales']

        return(predictions)

def construct_Pipline():
    date_transformer = X_Date_Transformer()
    baseline_model = Baseline_Model()
    pline = Pipeline([('date_transformer', date_transformer)
                                ,('baseline_model', baseline_model)])

    return(pline)


