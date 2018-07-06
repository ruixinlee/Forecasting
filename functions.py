import pandas as pd
import numpy as np
from dateutil import parser
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
import load_data



def separate_training_portfolio(full_dataset, target_vehicle_line, split_date ,timeline, vehicle_line):
    full_dataset = full_dataset.sort_values(by=timeline)
    selection = full_dataset[(full_dataset[vehicle_line] == target_vehicle_line)]
    selection.set_index(timeline, inplace = True)
    selection = selection.resample('MS').ffill()
    train = selection[(selection.index <= split_date)]
    portfolio = selection[(selection.index > split_date)]
    return (train, portfolio)





def split_train_test(sales_data_all, vl, actuals_end_date,test_start_date, timeline, vehicle_line):
    (df, portfolio_data) = separate_training_portfolio(sales_data_all, vl, actuals_end_date, timeline, vehicle_line)
    df['actual_sales_lag1'] = df['actual_sales'].shift(1)
    df['actual_sales_lag2'] = df['actual_sales'].shift(2)
    df['actual_sales_lag3'] = df['actual_sales'].shift(3)
    df.dropna(inplace=True)
    train = df[(df.index < test_start_date)]
    test = df[(df.index>= test_start_date)]
    return(train, test)


def build_output(test, pred_test, cols, model_name, vl):
    # Generate results
    results_dict = display_results(test['actual_sales'].values, pred_test, model_name)
    temp_res = pd.DataFrame(columns=cols)
    temp_res['Forecasted Month'] = test.index
    temp_res['Forecast'] = pred_test.values
    temp_res['actual_sales'] = test['actual_sales'].tolist()
    temp_res['Car Line (Model)'] = [vl] * temp_res.shape[0]
    temp_res['Forecasting Model'] = [results_dict['Model']] * temp_res.shape[0]
    temp_res['Forecasting Model Notes'] = ['Script: exo_lm_global_residuals_with_arima'] * temp_res.shape[0]
    temp_res['Country'] = ['GLOBAL'] * temp_res.shape[0]
    temp_res['RMSE'] = [results_dict['RMSE'] ** 2] * temp_res.shape[0]
    temp_res['TEASE'] = [results_dict['TAD']] * temp_res.shape[0]
    temp_res['TEASEP'] = [results_dict['TEASEP']] * temp_res.shape[0]
    temp_res['mape'] = [results_dict['MAPE']] * temp_res.shape[0]

    return (temp_res)


def plot_forecast_sales(actual_sales, predictions_all, predict_ci_upper ,predict_ci_lower, vl, nobs):

    fig, ax = plt.subplots(figsize=(16 ,10))
    ax.xaxis.grid()
    ax.yaxis.grid()
    idx = actual_sales.index
    ax.plot(actual_sales, 'k-' ,color = 'red', label = 'actual normalised sales')
    #     ax2 = ax.twinx()
    #     ax2.plot(data[['IHS_t']].iloc[skip:,:], 'k--',color = 'blue',  alpha=0.65, label = 'IHS segment share')


    ax.plot(idx[ :-nobs], predictions_all[ :-nobs], 'gray', label = 'in-sample one step ahead')
    ax.plot(idx[-(nobs +1):], predictions_all[-(nobs +1):], '--k' ,color = 'black', label = 'out-sample dynamic')
    ax.fill_between(idx, predict_ci_upper, predict_ci_lower, alpha=0.30, label = '95 percentile confidence interval')
    ax.legend()
    ax.set_title(f'SARIMAX sales forecasting for {vl}')



def root_mean_squared_error(y_true, y_pred):
    sum = 0
    nr = 0
    for i, elem in enumerate(y_true):
        if ((not math.isnan(y_true[i])) & (not math.isnan(y_pred[i]))):
            nr += 1
            sum += (y_true[i] - y_pred[i]) ** 2
    return math.sqrt(sum / nr) if (nr != 0) else 0


def mean_absolute_percentage_error(y_true, y_pred):
    sum = 0
    nr = 0
    for i, elem in enumerate(y_true):
        if ((not math.isnan(y_true[i])) & (not math.isnan(y_pred[i]))):
            if (y_true[i] != 0):
                nr += 1
                sum += (abs(y_true[i] - y_pred[i]) / y_true[i])
    return (sum / nr) * 100 if (nr != 0) else 0


def total_absolute_difference(y_true, y_pred):
    difference = y_pred - y_true
    return difference.sum()


def TEASEP(y_true, y_pred):
    difference = y_pred - y_true
    return (difference.sum() / y_true.sum()) if (y_true.sum() != 0) else 0


def display_results(true_values, predictions, name, vl=None):
    rmse = root_mean_squared_error(true_values, predictions)
    mape = mean_absolute_percentage_error(true_values, predictions)
    tad = total_absolute_difference(true_values, predictions)
    teasep = TEASEP(true_values, predictions)

    metrics_dict = {"Model": name,
                    "RMSE": rmse,
                    "MAPE": mape / 100,
                    "TAD": tad,
                    "TEASEP": teasep,
                    "vehicle_line": vl
                    }

    return (metrics_dict)


def plot_forecast_sales(actual_sales, predictions_all, predict_ci_upper, predict_ci_lower, vl, nobs):
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.xaxis.grid()
    ax.yaxis.grid()
    idx = actual_sales.index
    ax.plot(actual_sales, 'k-', color='red', label='actual sales')

    ax.plot(idx[:-nobs + 1], predictions_all[:-nobs + 1], 'gray', label='in-sample one step ahead')
    ax.plot(idx[-(nobs):], predictions_all[-(nobs):], 'k-', color='black', label='out-sample dynamic')
    ax.fill_between(idx[-(nobs):], predict_ci_upper[-(nobs):], predict_ci_lower[-(nobs):], alpha=0.30,
                    label='90 percentile confidence interval')
    ax.legend()
    ax.set_title(f'Ensemble forecasting for {vl}')
