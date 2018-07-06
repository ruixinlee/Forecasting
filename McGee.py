
# coding: utf-8

# # Model McGee

# ## The model
# 
# 
# We assume that $y_t$ follows the following model:
# 
# $$y_t =  \beta x_t +\epsilon_t$$  (1)
# 
# where 
# 
# $y_t$, total sales in month t
# 
# $x_t$, IHS vehicle line or segment quarterly sales in month t
# 
# We assume that $\epsilon_t$ follows an SARIMA process, i.e. $\phi_p(L)\Phi_P(L)\delta_d(L)\Delta_D(L)\epsilon_t = \theta_q(L)\Theta_Q(L)\zeta_t$ where $\zeta_t$ is an i.i.d white noise.
# 
# If the inverse operators, $\phi_p^{-1}(L)$, $\Phi_P^{-1}(L)$, $\delta_d^{-1}(L)$, $\Delta_D^{-1}(L)$, exists, we can denote $$\Gamma(L) = \Delta_D^{-1}(L)\delta_d^{-1}(L)\Phi_P^{-1}(L)\phi_p^{-1}(L)\theta_q(L)\Theta_Q(L)$$
# 
# and rewrite (1) as 
# 
# $$y_t =  \beta x_t + \Gamma(L)\zeta_t$$
# 
# 
# We explicitly assume that the model intercept is 0. This is also supported by the fact that the intercept has insignificant p-values when fitted. 
# This becomes a linear regression between $x_t$ and $y_t$ by treating its residuals as a function of ARIMA
# 
# ## Citations
# https://newonlinecourses.science.psu.edu/stat510/node/72/
# 
# 
# 
# 
# 
# 
# 

# ## Model parameterisation
# 
# * for the exogenous variable $x_t$, the algorithm chooses between IHS segement and IHS quarter sales by maximising the in-sample $R^2$
# 
# 
# * we parameterise the model in two steps. First we fit $\beta$'s to the exogenous variable. Then we fit an ARIMA process to the residuals $\epsilon_t$
# 
# 
# * to ensure we capture all information about $\beta$'s, we fit t $\beta$'s too the full length of history. 
# 
# 
# * for the residuals $\epsilon_t$, we fit an SARIMA to at most three years of history in our case 2014-04-01 till 2017-03-31. This is under the assumption that older history is no loner relevant to today's world. 
# 
# 
# * we designed an algorithm to determine the struture of SARIMA and trend by looking at the in-sample PACF and p-values for each series. This algorithm is rule based - using these steps: https://people.duke.edu/~rnau/arimrule.htm
# 
# 
# * we use statsmodel api to calibrate the parameters. 
# 
# ## Projections and model evaluations
# * we projected 12 steps ahead and evaluate the model using MAPE, teasep, tease and RMSE.
# 
# ## Caveats
# * even though we have taken steps to avoid overfitting by parameterising only using in-sample data, we still need to do a rolling window test on this.
# 
# 
# * this model has yet to be reviewed.
# 

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
from dateutil import parser
import math
import pylab
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn import linear_model
import statsmodels.api as sm
import load_data
import datetime as dt


# In[2]:


# Read data and set parameters
timeline = 'target_month'
vehicle_line = 'vehicle_line'
actuals_end_date = parser.parse('2018-04-01')
test_start_date = parser.parse('2017-04-01')

all_predictors = ['ihs_t_vl', 'IHS_t']
target = 'actual_sales'
model_name = 'model mcgee v5'


sales_table = load_data.import_sales_data_from_bq()
sales_data_all = load_data.import_sales_data_from_bq()
sales_data_all[timeline] = pd.to_datetime(sales_data_all[timeline])
sales_data_all['REL'] = sales_data_all['ihs_t_vl'] / (sales_data_all['IHS_t']+1)
#sales_data_all = sales_data_all[sales_data_all[timeline]>= '2015-04-01 ']
sales_data_all['zeros']  = 0


# In[3]:


def separate_training_portfolio(full_dataset, target_vehicle_line, split_date):
    full_dataset = full_dataset.sort_values(by=timeline)
    selection = full_dataset[(full_dataset[vehicle_line] == target_vehicle_line)] 
    selection.set_index(timeline, inplace = True)
    selection = selection.resample('M').ffill()
    train = selection[(selection.index <= split_date)]
    portfolio = selection[(selection.index > split_date)]
    return (train, portfolio)


# In[4]:


def split_train_test(sales_data_all, vl, actuals_end_date):
    (df, portfolio_data) = separate_training_portfolio(sales_data_all, vl, actuals_end_date)
    df['actual_sales_lag1'] = df['actual_sales'].shift(1)
    df['actual_sales_lag2'] = df['actual_sales'].shift(2)
    df['actual_sales_lag3'] = df['actual_sales'].shift(3)
    df.dropna(inplace=True)
    train = df[(df.index<test_start_date)]
    test = df[(df.index>=test_start_date)]
    return(train, test)


# In[5]:


def plot_forecast_sales(actual_sales, predictions_all, predict_ci_upper,predict_ci_lower, vl, nobs):

    fig, ax = plt.subplots(figsize=(16,10))
    ax.xaxis.grid()
    ax.yaxis.grid()
    idx = actual_sales.index
    ax.plot(actual_sales, 'k.',color = 'red', label = 'actual normalised sales')
    #     ax2 = ax.twinx()
    #     ax2.plot(data[['IHS_t']].iloc[skip:,:], 'k--',color = 'blue',  alpha=0.65, label = 'IHS segment share')

    ax.plot(idx[ :-nobs], predictions_all[ :-nobs], 'gray', label = 'in-sample one step ahead')
    ax.plot(idx[-(nobs-1):], predictions_all[-(nobs-1):], 'k-',color = 'black', label = 'out-sample dynamic')
    ax.fill_between(idx, predict_ci_upper, predict_ci_lower, alpha=0.30, label = '95 percentile confidence interval')
    ax.legend()
    ax.set_title(f'SARIMAX sales forecasting for {vl}')
    
    


# In[6]:


def calibrate_SARIMAX(res,sarimax):
    print(sarimax)
    model = sm.tsa.statespace.SARIMAX(res, order = sarimax['order'], seasonal_order = sarimax['sorder'], 
                                              trend = sarimax['trend'], enforce_stationarity = sarimax['es'] , mle_regression=False).fit()
    return(model)

def calibrate_pacf(res):
    pacf = sm.tsa.stattools.pacf(res, method='ywmle', alpha = 0.5)
    return(pacf)


# In[7]:


# Fit and predict using a linear model using statsmodels
def fit_and_predict_sm(train, test, target_attr, predictor_attr, plot_results = True, sarimax =None, es =True, calibrate = False):
    res_calibrate_date = '2014-04-01'
    train = train.dropna()
    
    y = train[target_attr]
    
    df_all = pd.concat([train,test])
    df_all = df_all[df_all.index>=res_calibrate_date]
    
    vl = train['vehicle_line'].tolist()[0]
    
    nobs = test.shape[0]
    R_2_list = []
    
    for p in predictor_attr:
        X = train[p]
        model = sm.OLS(y, X).fit()
        R_2_list.append(model.rsquared)
    
    print(predictor_attr)
    print(R_2_list)
    selected_attr = [predictor_attr[i] for i,p in enumerate(R_2_list) if p == max(R_2_list)]
    X = train[selected_attr]
    model = sm.OLS(y, X).fit()
    res_ori = model.resid
    
    
    ##test residuals
    lj_lbvalue,lj_pvalue = sm.stats.diagnostic.acorr_ljungbox(res_ori)
    failed_lag = lj_pvalue <= 0.05
    failed_lag_pos = np.array(range(len(failed_lag)))[failed_lag]
    print('failed lj lags are at ' + ','.join(str(i) for i in failed_lag_pos) 
          + ', with values ' + ','.join(str(i) for i in lj_pvalue[failed_lag_pos]))

    ########    
    X_all = df_all[selected_attr]
    ols_predictions = model.get_prediction(X_all) 
    predictions_ols = ols_predictions.predicted_mean 
    predictions_all = ols_predictions.predicted_mean 
    predict_ci = ols_predictions.conf_int(alpha=0.5)
    predict_ci_upper = predict_ci[:,1] 
    predict_ci_lower = predict_ci[:,0] 
    res = res_ori[res_ori.index>=res_calibrate_date] 

    if plot_results:
        print(model.summary())
        plot_acf_pacf(res)
    ########
    print('start')
    if calibrate:
        sarimax = {}
        
        if calibrate:
            sarimax = {'order' : (0,0,0),
                       'sorder': (0,0,0,12),
                       'trend' : [0,0,0,0],
                       'es':  False}
            
#############calibrating Seasonal AR
            print('acf12')
            print(calibrate_pacf(res)[0][12])
            if calibrate_pacf(res)[0][12] >= 0.2:
                if calibrate_pacf(res)[0][24] > 0.05:
                    sarimax['sorder'] = (0,1,0,12)
                    res_model = calibrate_SARIMAX(res,sarimax)
                    
                    if calibrate_pacf(res_model.resid)[0][12] >= 0.3:
                        sarimax['sorder'] = (1,1,0,12)
                        res_model = calibrate_SARIMAX(res_model.resid,sarimax)
                else:
                    sarimax['sorder'] = (1,0,0,12)
                    res_model = calibrate_SARIMAX(res,sarimax)
                    
                    if calibrate_pacf(res_model.resid)[0][12] < -0.2:
                        sarimax['sorder'] = (0,0,0,12)
                        res_model = calibrate_SARIMAX(res_model.resid,sarimax)
            else:
                sarimax['sorder'] = (1,0,0,12)
                res_model = calibrate_SARIMAX(res,sarimax)
            
            ###calibrating AR
            if calibrate_pacf(res_model.resid)[0][1] >= 0.3:
                
                if (calibrate_pacf(res_model.resid)[0][2]>0.2 and calibrate_pacf(res_model.resid)[0][3]>0.1):
                    
                    sarimax['order'] = (0,1,0)
                    res_model = calibrate_SARIMAX(res,sarimax)
                    if (calibrate_pacf(res_model.resid)[0][1] > 0.3):
                        sarimax['order'] = (1,1,0)
                        res_model = calibrate_SARIMAX(res,sarimax)                       
                    elif (calibrate_pacf(res_model.resid)[0][1] < -0.35):
                        sarimax['order'] = (0,1,1)
                        res_model = calibrate_SARIMAX(res,sarimax)                       
                    elif (calibrate_pacf(res_model.resid)[0][1] > -0.35 and calibrate_pacf(res_model.resid)[0][1] <=0):
                            sarimax['order'] = (0,0,0)
                            res_model = calibrate_SARIMAX(res,sarimax)                       
                        
                else:
                    sarimax['order'] = (1,0,0)
                    res_model = calibrate_SARIMAX(res,sarimax)

            print(res_model.summary())                                                                         
            print('fitting trend')

#                 sarimax['trend'] = [0,0,1,0]

            #fit a cubic for curves with little parameters, cubic because of the change in cyclical change in trend
########################################
            sum_of_params = sum(sarimax['order'] ) + sum(sarimax['sorder'] )
            if sum_of_params<=13:
                trend = [[0,0,1,0],[0,0,0,1]]
                sarimax['trend'] = [0,0,0,0]
                aic = []
                p_values = []
                for i,t in enumerate(trend):
                    print('fitting t')
          
                    res_model = sm.tsa.statespace.SARIMAX(res, order = sarimax['order'], 
                                                          seasonal_order = sarimax['sorder'], 
                                                          trend = t, 
                                                          enforce_stationarity = sarimax['es'], 
                                                          mle_regression=False).fit()
                    aic.append(res_model.aic)
                    p_values.append(res_model.pvalues[0])

                print('p_values')
                print(p_values)
                print('aic')
                print(aic)

                trend_selected = [(k,trend[k]) for k,a in enumerate(aic) if a==min(aic)][0]
                print(trend_selected[0])
                if p_values[trend_selected[0]] < 0.2:
                    sarimax['trend'] = trend_selected[1]
                
        
#######################################            
    print('fitting model')
    
    
    res_model = sm.tsa.statespace.SARIMAX(res,order = sarimax['order'], 
                                                  seasonal_order = sarimax['sorder'], 
                                                  trend = sarimax['trend'], 
                                                  enforce_stationarity = sarimax['es'], mle_regression=False).fit()
    print(sarimax)
    print(res_model.summary())
    if plot_results: 
#             print(res_model.summary())
        plot_acf_pacf(res_model.resid)
        #res_model.plot_diagnostics()    
    predict_res = res_model.get_prediction(start  = X_all.index.min(),end = X_all.index.max())


    predict_res_mean = predict_res.predicted_mean
    predict_ci = predict_res.conf_int(alpha=0.5)
    predict_ci_upper = predict_ci.iloc[:,1] + predictions_ols
    predict_ci_lower =  predict_ci.iloc[:,0] + predictions_ols
    predictions_all =  predict_res_mean + predictions_ols

    
    plot_forecast_sales(df_all[target_attr], predictions_all, predict_ci_upper,predict_ci_lower, vl, nobs)
    predictions_test = predictions_all.iloc[-nobs:]
    
    return predictions_test


# In[8]:


# Fit and predict using a linear model from sklearn 
def fit_and_predict_lm(train, test, target_attr, predictor_attr):
  train = train.dropna()
  X = train[predictor_attr]
  y = train[target_attr]
  lm = linear_model.LinearRegression()
  model = lm.fit(X,y)
  #print(lm.summary())
  X_test = test[predictor_attr]
  predictions_test = lm.predict(X_test)
  return predictions_test


# In[9]:


# Plot predictions
def plot_predictions(sales_vl, predicted_values, vl):
  fig, ax = plt.subplots(figsize=(16,10))
  ax.xaxis.grid()
  ax.yaxis.grid()
  ax.plot(sales_vl[['actual_sales']], 'k--',color = 'red', label = 'Actual Sales')
  ax2 = ax.twinx()
  ax2.plot(sales_vl[['REL']], 'k--',color = 'blue',  alpha=0.15, label = 'Segment Share')
  ax.plot(predicted_values, 'k--', linestyle='--', linewidth=2, label = 'Forecast')
  ax.legend()
  ax.set_title("Forecast for " + vl)


# In[10]:


def plot_acf_pacf(res):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(res, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(res, method = 'ywmle', ax=ax2)
    plt.show()


# In[11]:


# Evaluation metrics
# RMSE
def root_mean_squared_error(y_true, y_pred):
  sum = 0
  nr = 0
  for i, elem in enumerate(y_true):
      if ((not math.isnan(y_true[i])) & (not math.isnan(y_pred[i]))):
          nr += 1
          sum += (y_true[i] - y_pred[i]) ** 2
  return math.sqrt(sum / nr) if (nr != 0) else 0


# MAPE
def mean_absolute_percentage_error(y_true, y_pred):
  sum = 0
  nr = 0
  for i, elem in enumerate(y_true):
      if ((not math.isnan(y_true[i])) & (not math.isnan(y_pred[i]))):
          if (y_true[i] != 0):
              nr += 1
              sum += (abs(y_true[i] - y_pred[i]) / y_true[i])
  return (sum / nr) * 100 if (nr != 0) else 0

# TEASE
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

    metrics_dict = {"Model":name,
                    "RMSE": rmse,
                    "MAPE": mape,
                    "TAD":tad,
                    "TEASEP": teasep,
                    "vehicle_line":vl
                   }

    return (metrics_dict)


# In[12]:


vl = 'DISCOVERY SPORT'
metrics_dat = []
sarimax_order = {}
sarimax_order['order'] = (0, 1, 0)
sarimax_order['sorder'] = (1, 0 , 0, 12)
sarimax_order['trend'] = [0, 0, 0, 0]
sarimax_order['es'] = False

train,test = split_train_test(sales_data_all, vl, actuals_end_date)

pred_test = fit_and_predict_sm(train, test, target, all_predictors, plot_results = True, sarimax = sarimax_order, calibrate = False)
metrics_dat.append(display_results(test[target].values, pred_test, model_name))
metrics_dat = pd.DataFrame.from_dict(metrics_dat)

print(metrics_dat)




# In[13]:


vl = 'DISCOVERY'

metrics_dat = []
sarimax_order = {}
sarimax_order['order'] = (0, 0, 0)
sarimax_order['sorder'] = (1, 0 , 0, 12)
sarimax_order['trend'] = [0, 0, 0, 0]
sarimax_order['es'] = False

train,test = split_train_test(sales_data_all, vl, actuals_end_date)

pred_test = fit_and_predict_sm(train, test, target, all_predictors, plot_results = True, sarimax = sarimax_order, calibrate = False)
metrics_dat.append(display_results(test[target].values, pred_test, model_name))
metrics_dat = pd.DataFrame.from_dict(metrics_dat)

print(metrics_dat)





# In[14]:


vl = 'RANGE ROVER SPORT'
metrics_dat = []
sarimax_order = {}
sarimax_order['order'] = (0, 0, 0)
sarimax_order['sorder'] = (1, 0 , 0, 12)
sarimax_order['trend'] = [0, 0, 1, 0]
sarimax_order['es'] = False

train,test = split_train_test(sales_data_all, vl, actuals_end_date)

pred_test = fit_and_predict_sm(train, test, target, all_predictors, plot_results = True, sarimax = sarimax_order, calibrate = True)
metrics_dat.append(display_results(test[target].values, pred_test, model_name))
metrics_dat = pd.DataFrame.from_dict(metrics_dat)

print(metrics_dat)



# In[15]:


vl = 'RANGE ROVER EVOQUE'

metrics_dat = []
sarimax_order = [(1, 1, 1), (1, 0, 0, 12), [0, 1, 0, 0], False]
train,test = split_train_test(sales_data_all, vl, actuals_end_date)

pred_test = fit_and_predict_sm(train, test, target, all_predictors, plot_results = False, sarimax = sarimax_order, calibrate = True)
metrics_dat.append(display_results(test[target].values, pred_test, model_name))
metrics_dat = pd.DataFrame.from_dict(metrics_dat)

print(metrics_dat)


# In[16]:


vl = 'XF'

metrics_dat = []
sarimax_order = [(1,0,0),(1,1,0,12), [0,0,0,0]]
train,test = split_train_test(sales_data_all, vl, actuals_end_date)

pred_test = fit_and_predict_sm(train, test, target, all_predictors, plot_results = False, sarimax = sarimax_order, calibrate = True)
metrics_dat.append(display_results(test[target].values, pred_test, model_name))
metrics_dat = pd.DataFrame.from_dict(metrics_dat)

print(metrics_dat)


# In[17]:


vl = 'XJ'

metrics_dat = []
sarimax_order = {}
sarimax_order['order'] = (0, 0, 0)
sarimax_order['sorder'] = (1, 0, 0, 12)
sarimax_order['trend'] = [0, 0, 0, 0]
sarimax_order['es'] = False

train,test = split_train_test(sales_data_all, vl, actuals_end_date)
pred_test = fit_and_predict_sm(train, test, target, all_predictors, plot_results = True, sarimax = sarimax_order, calibrate = True)
metrics_dat.append(display_results(test[target].values, pred_test, model_name))
metrics_dat = pd.DataFrame.from_dict(metrics_dat)

print(metrics_dat)


# In[18]:


vl = 'F-TYPE'
metrics_dat = []
sarimax_order = [(1,0,0),(1,1,0,12), [0,0,0,0]]
train,test = split_train_test(sales_data_all, vl, actuals_end_date)

pred_test = fit_and_predict_sm(train, test, target, all_predictors, plot_results = False, sarimax = sarimax_order, calibrate = True)
metrics_dat.append(display_results(test[target].values, pred_test, model_name))
metrics_dat = pd.DataFrame.from_dict(metrics_dat)

print(metrics_dat)


# In[19]:


vl = 'RANGE ROVER'
metrics_dat = []
sarimax_order = {}
sarimax_order['order'] = (0, 0, 0)
sarimax_order['sorder'] = (1, 0 , 0, 12)
sarimax_order['trend'] = [0, 0, 1, 0]
sarimax_order['es'] = False

train,test = split_train_test(sales_data_all, vl, actuals_end_date)

pred_test = fit_and_predict_sm(train, test, target, all_predictors, plot_results = True, sarimax = sarimax_order, calibrate = False)
metrics_dat.append(display_results(test[target].values, pred_test, model_name))
metrics_dat = pd.DataFrame.from_dict(metrics_dat)

print(metrics_dat)


# In[20]:


vl = 'XE'


metrics_dat = []
sarimax_order = {}
sarimax_order['order'] = (0, 0, 0)
sarimax_order['sorder'] = (1, 0 , 0, 12)
sarimax_order['trend'] = [0, 1, 0 ,0]
sarimax_order['es'] = False

train,test = split_train_test(sales_data_all, vl, actuals_end_date)
pred_test = fit_and_predict_sm(train, test, target, all_predictors, plot_results = True, sarimax = sarimax_order, calibrate = False)
metrics_dat.append(display_results(test[target].values, pred_test, model_name))
metrics_dat = pd.DataFrame.from_dict(metrics_dat)

print(metrics_dat)


# In[21]:


train.head()


# In[22]:


cols = ['TimeStamp','Forecasting Model','Forecasting Model Notes','Forecasted Month','Car Line (Model)',
        'Forecast','Country','MSE','TEASE','TEASEP','mape']


# In[23]:


def build_output(test,pred_test, cols, model_name,vl ):
    # Generate results
    results_dict = display_results(test['actual_sales'].values, pred_test,model_name)
    temp_res = pd.DataFrame(columns=cols)
    temp_res['Forecasted Month'] = test.index
    temp_res['Forecast'] = pred_test.values
    temp_res['Car Line (Model)'] = [vl] * temp_res.shape[0]
    temp_res['Forecasting Model'] = [results_dict['Model']] * temp_res.shape[0]
    temp_res['Forecasting Model Notes'] = ['Script: exo_lm_global_residuals_with_arima'] * temp_res.shape[0]
    temp_res['Country'] = ['GLOBAL'] * temp_res.shape[0]
    temp_res['MSE'] = [results_dict['RMSE']**2] * temp_res.shape[0]
    temp_res['TEASE'] = [results_dict['TAD']] * temp_res.shape[0]
    temp_res['TEASEP'] = [results_dict['TEASEP']] * temp_res.shape[0]
    temp_res['mape'] = [results_dict['MAPE']] * temp_res.shape[0]
    return(temp_res)


# In[24]:


LM_all_arima = {'DISCOVERY SPORT':[(1,0,0),(1,1,0,12), [0,0,0,0]] ,
                'DISCOVERY': [(1,0,0),(1,1,0,12), [0,0,0,0]],
                'RANGE ROVER SPORT':[(1,0,0),(1,1,0,12), [0,0,0,0]],
                'RANGE ROVER EVOQUE':[(1,0,0),(1,1,0,12), [0,0,0,0]],
                'XF':[(1,0,0),(1,1,0,12), [0,0,0,0]],
                'XJ': [(1,0,0),(1,1,0,12), [0,0,0,0]],
                'F-TYPE':[(1,0,0),(1,1,0,12), [0,0,0,0]],
                'RANGE ROVER': [(1,0,0),(1,1,0,12), [0,0,0,0]],
                'XE':[(1, 0, 0), (1, 0, 0, 12), [0, 1, 0, 0]]}


final_result = []
models_performance = []

csv_output_name = f'.\\output\\{model_name}'
for vl in sales_data_all[vehicle_line].unique():
  if (vl not in ['E-PACE','F-PACE','I-PACE','RANGE ROVER VELAR','XK','FREELANDER','DEFENDER']):
    print("\n"+vl)
    sarimax_order = LM_all_arima[vl]
    df, _= separate_training_portfolio(sales_data_all, vl, actuals_end_date)
    train, test = split_train_test(sales_data_all, vl, actuals_end_date)
    pred_test = fit_and_predict_sm(train, test, 'actual_sales', all_predictors, plot_results = False, 
                                     sarimax =  sarimax_order, calibrate = True )
    #plot_predictions(df, pred_test, vl)
    temp_result = build_output(test, pred_test, cols, model_name,vl )
    model_perf=display_results(test[target], pred_test, model_name, vl)
    models_performance.append(model_perf)
    final_result.append(temp_result) 
    
final_result = pd.concat(final_result)
final_result = final_result[cols]
now = pd.Timestamp.now().strftime('%Y-%m-%d')
final_result['TimeStamp'] =now
final_result.to_csv(f".\\{csv_output_name}.csv") 

models_performance = pd.DataFrame.from_dict(models_performance)
print(models_performance)


# In[25]:


print(models_performance[['MAPE','RMSE','TEASEP','vehicle_line']].sort_values(by = 'vehicle_line'))

