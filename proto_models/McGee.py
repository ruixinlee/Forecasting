import pandas as pd
import numpy as np
from dateutil import parser
import matplotlib.pyplot as plt
import statsmodels.api as sm
import load_data
import sys
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline

sys.path.append('..')
import functions


def plot_acf_pacf(res):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(res, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(res, method='ywmle', ax=ax2)
    plt.show()


class IHS_selector(BaseEstimator,TransformerMixin):
    def __init__(self, predictors = ['ihs_t_vl', 'IHS_t']):
        self.predictors = predictors

    def fit(self, X, y=None):
        R_2_list = []

        for p in self.predictors:
            X1 = X[p]
            model = sm.OLS(y, X1).fit()
            R_2_list.append(model.rsquared)
        self.selected_predictor = [self.predictors[i] for i, p in enumerate(R_2_list) if p == max(R_2_list)]
        print(self.selected_predictor)
        return(self)

    def transform(self, X):
        X = X[self.selected_predictor]
        return(X)


class McGee(BaseEstimator,RegressorMixin):
    def __init__(self, sarimax_params = None, print = False, calibrate = True, target_attr = 'actual_sales',predictor_attr = None, vl = None ):
        self.params = None
        self.print = print
        self.calibrate = calibrate
        self.target_attr = target_attr
        self.predictor_attr = predictor_attr
        self.vl = vl
        if calibrate:
            if sarimax_params is  not None:
                print('Given sarimax_params will be ignored for calibration')
            self.sarimax_params = None,
        else:
            self.sarimax_params = sarimax_params

    def calibrate_SARIMAX(self, ts, sarimax):
        model = sm.tsa.statespace.SARIMAX(ts, order=sarimax['order'], seasonal_order=sarimax['sorder'],
                                          trend=sarimax['trend'], enforce_stationarity=sarimax['es'],
                                          mle_regression=False).fit()
        return (model)

    def calibrate_pacf(self, res):
        pacf = sm.tsa.stattools.pacf(res, method='ywmle', alpha=0.95)
        return (pacf)

    def calibrate_sarimax_structure(self, res):
        sarimax = {'order': (0, 0, 0),
                   'sorder': (0, 0, 0, 12),
                   'trend': [0, 0, 0, 0],
                   'es': False}

        if self.calibrate_pacf(res)[0][12] >= 0.2:
            if self.calibrate_pacf(res)[0][24] > 0.05:
                sarimax['sorder'] = (0, 1, 0, 12)
                res_model = self.calibrate_SARIMAX(res, sarimax)

                if self.calibrate_pacf(res_model.resid)[0][12] >= 0.3:
                    sarimax['sorder'] = (1, 1, 0, 12)
                    res_model = self.calibrate_SARIMAX(res_model.resid, sarimax)
            else:
                sarimax['sorder'] = (1, 0, 0, 12)
                res_model = self.calibrate_SARIMAX(res, sarimax)

                if self.calibrate_pacf(res_model.resid)[0][12] < -0.2:
                    sarimax['sorder'] = (0, 0, 0, 12)
                    res_model = self.calibrate_SARIMAX(res_model.resid, sarimax)
        else:
            sarimax['sorder'] = (1, 0, 0, 12)
            res_model = self.calibrate_SARIMAX(res, sarimax)

        if self.calibrate_pacf(res_model.resid)[0][1] >= 0.3:

            if (self.calibrate_pacf(res_model.resid)[0][2] > 0.2 and self.calibrate_pacf(res_model.resid)[0][3] > 0.1):

                sarimax['order'] = (0, 1, 0)
                res_model = self.calibrate_SARIMAX(res, sarimax)
                if (self.calibrate_pacf(res_model.resid)[0][1] > 0.3):
                    sarimax['order'] = (1, 1, 0)
                    res_model = self.calibrate_SARIMAX(res, sarimax)
                elif (self.calibrate_pacf(res_model.resid)[0][1] < -0.35):
                    sarimax['order'] = (0, 1, 1)
                    res_model = self.calibrate_SARIMAX(res, sarimax)
                elif (self.calibrate_pacf(res_model.resid)[0][1] > -0.35 and self.calibrate_pacf(res_model.resid)[0][1] <= 0):
                    sarimax['order'] = (0, 0, 0)
                    res_model = self.calibrate_SARIMAX(res, sarimax)

            else:
                sarimax['order'] = (1, 0, 0)
                res_model = self.calibrate_SARIMAX(res, sarimax)

      ########################################
        sum_of_params = sum(sarimax['order']) + sum(sarimax['sorder'])
        if sum_of_params <= 13:
            trend = [[0, 0, 1, 0], [0, 0, 0, 1]]
            aic = []
            p_values = []
            sarimax_temp = sarimax.copy()
            for i, t in enumerate(trend):
                print('fitting t')
                sarimax_temp['trend'] = t
                res_model =self.calibrate_SARIMAX(res, sarimax_temp)
                aic.append(res_model.aic)
                p_values.append(res_model.pvalues[0])

            trend_selected = [(k, trend[k]) for k, a in enumerate(aic) if a == min(aic)][0]

            if p_values[trend_selected[0]] < 0.2:
                sarimax['trend'] = trend_selected[1]

        self.sarimax_structure = sarimax
        return(self.sarimax_structure)

    def fit(self, X, y):
        self.res_calibrate_date = X.index.max() - pd.DateOffset(years=3)
        self.ols_model = sm.OLS(y, X).fit()
        res_ori = self.ols_model.resid
        res = res_ori[res_ori.index > self.res_calibrate_date]
        if self.calibrate:
            self.sarimax_structure = self.calibrate_sarimax_structure(res)
        self.res_model = self.calibrate_SARIMAX(res, self.sarimax_structure )
        self.fittedvalues = self.predict(X[X.index > self.res_calibrate_date],out_sample=False)

    def predict(self,X, out_sample = True):
        ols_prediction = self.ols_model.get_prediction(X)
        res_prediction = self.res_model.get_prediction(start=X.index.min(), end=X.index.max())

        ols_prediction_mean = ols_prediction.predicted_mean
        res_prediction_mean = res_prediction.predicted_mean

        predictions_all = res_prediction_mean + ols_prediction_mean

        if out_sample:
            predict_ci = res_prediction.conf_int(alpha=0.05)
            self.upperpi =predict_ci.iloc[:, 1] + ols_prediction_mean
            self.lowerpi = predict_ci.iloc[:, 0] + ols_prediction_mean

            self.upperpi[self.upperpi < 0] = 0
            self.lowerpi[self.lowerpi < 0] = 0

        return (predictions_all)





if __name__ == '__main__':

    timeline = 'target_month'
    vehicle_line = 'vehicle_line'

    actuals_end_date = parser.parse('2018-04-01')
    test_start_date = parser.parse('2017-04-01')

    all_predictors = ['ihs_t_vl', 'IHS_t']
    target = 'actual_sales'
    model_name = 'model mcgee v5'

    sales_data_all = load_data.import_sales_data_from_bq()
    sales_data_all[timeline] = pd.to_datetime(sales_data_all[timeline])
    sales_data_all['REL'] = sales_data_all['ihs_t_vl'] / (sales_data_all['IHS_t'] + 1)
    sales_data_all['zeros'] = 0

    cols = ['TimeStamp', 'Forecasting Model', 'Forecasting Model Notes', 'Forecasted Month', 'Car Line (Model)',
            'Forecast', 'Country', 'MSE', 'TEASE', 'TEASEP', 'mape']

    ##individual car
    vl = 'XE'
    df, _ = functions.separate_training_portfolio(sales_data_all, vl, actuals_end_date, timeline, vehicle_line)
    train, test = functions.split_train_test(sales_data_all, vl, actuals_end_date, test_start_date, timeline,
                                             vehicle_line)

    pipeline = Pipeline([('IHS_selector', IHS_selector())
                            , ('McGee', McGee(calibrate=True, vl=vl))])

    pipeline.fit(X=train, y=train[target])
    out_sample_pred = pipeline.predict(X=test)
    in_sample_pred = pipeline.named_steps['McGee'].fittedvalues
    upper_pi = pipeline.named_steps['McGee'].upperpi
    lower_pi = pipeline.named_steps['McGee'].lowerpi
    model_perf = functions.display_results(test[target], out_sample_pred, model_name, vl)
    print(model_perf)



    final_result = []
    models_performance = []

    csv_output_name = f'..\\output\\{model_name}'
    for vl in sales_data_all[vehicle_line].unique():
        if (vl not in ['E-PACE', 'F-PACE', 'I-PACE', 'RANGE ROVER VELAR', 'XK', 'FREELANDER', 'DEFENDER']):
            print("\n" + vl)
            df, _ = functions.separate_training_portfolio(sales_data_all, vl, actuals_end_date, timeline, vehicle_line)
            train, test = functions.split_train_test(sales_data_all, vl, actuals_end_date, test_start_date, timeline, vehicle_line)

            pipeline = Pipeline([('IHS_selector', IHS_selector())
                                 , ('McGee', McGee(calibrate=True, vl=vl))])

            pipeline.fit(X=train,  y=train[target])
            out_sample_pred =  pipeline.predict(X=test)
            in_sample_pred = pipeline.named_steps['McGee'].fittedvalues
            upper_pi = pipeline.named_steps['McGee'].upperpi
            lower_pi = pipeline.named_steps['McGee'].lowerpi
            temp_result = functions.build_output(test, out_sample_pred, cols, model_name, vl)
            model_perf = functions.display_results(test[target], out_sample_pred, model_name, vl)
            models_performance.append(model_perf)
            final_result.append(temp_result)

    print(in_sample_pred)
    final_result = pd.concat(final_result)
    final_result = final_result[cols]
    now = pd.Timestamp.now().strftime('%Y-%m-%d')
    final_result['TimeStamp'] = now
    final_result.to_csv(f"{csv_output_name}.csv")

    models_performance = pd.DataFrame.from_dict(models_performance)
    print(models_performance)

    print(models_performance[['MAPE', 'RMSE', 'TEASEP', 'vehicle_line']])
    print('test')

