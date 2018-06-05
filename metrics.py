from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import csv
import os
from datetime import datetime as dt


metrics_csv = '.\\output\\model_metrics.csv'
col_calender = 'target_month'
col_market = 'market'
col_vehic = 'vehicle_line'
metrics_list = ['ad_r2', 'r2','rmse','mape']
def adjusted_r2_score(y_true, y_pred, n, p):
    #n = sample size
    #p = number of explainatory variables
    r2 = r2_score(y_true,y_pred)

    adjusted_r2 = (1-(1 - r2)* (n-1) /(n-p-1))
    return (adjusted_r2)

def mean_percentage_absolute_error(y_true, y_pred):
    n = len(y_true)
    dev_ratio = abs(y_true - y_pred)/y_true
    mape = dev_ratio.sum()/n*100
    return(mape)

def TEASE(y_true, y_pred):
    tease = sum(y_true)-sum(y_pred)
    return(tease)

def cal_metrics(test_target,predictions, data_size,  predictors_num=None):
    if predictors_num is not None:
        ad_r2 = adjusted_r2_score(test_target,predictions, data_size, predictors_num)
    else:
        ad_r2 = None
    r2 = r2_score(test_target, predictions)
    rmse = np.sqrt(mean_squared_error(test_target, predictions))
    mape = mean_percentage_absolute_error(test_target, predictions)
    tease = TEASE(test_target,predictions)
    md = {'ad_r2':ad_r2, 'r2': r2, 'rmse':rmse, 'mape': mape, 'TEASE':tease}
    return(md)

def write_metrics(row):
    csv_row = list(row.values())
    with open(metrics_csv, 'a') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writer.writerow(csv_row)


def initiate_csv(row):
    header = list(row.keys())
    if not os.path.isfile(metrics_csv):
        with open(metrics_csv,'w') as file:
            writer = csv.writer(file,delimiter = ',', lineterminator = '\n')
            writer.writerow(header)

def extract_pipeline(pipeline,model_name, predictors_num):
    if pipeline is not None and model_name is None:
        model_name = '-'.join(list(pipeline.named_steps.keys()))

        p_attr = list(pipeline.named_steps.values())[-1].__dict__.keys()
        if ('predictors_num' in p_attr) and (predictors_num is None):
            predictors_num =list(pipeline.named_steps.values())[-1].__dict__['predictors_num']

    return(model_name, predictors_num)

def create_row(test_X, test_round, pipeline, model_name, predictors_num, model_py, time):
    row = {}
    row['timestamp'] = dt.strftime(dt.now(), '%d-%m-%Y %H:%M')
    row['comp_name'] = os.environ['COMPUTERNAME']
    row['execution_time'] = time
    row['model_py'] = model_py
    if test_X is not None:
        row['data_size'] = test_X.shape[0]
        row['test_start'] = test_X[col_calender].min().strftime('%d-%m-%Y')
        row['test_end'] = test_X[col_calender].max().strftime('%d-%m-%Y')
    else:
        row['data_size'] = None
        row['test_start'] = None
        row['test_end'] = None
    row['test_round'] = test_round
    row['model_name'], row['predictors_num'] = extract_pipeline(pipeline,model_name,predictors_num)
    row['market'] = None
    row['vehicle_line'] = None

    for i in metrics_list:
        row[i] = None

    return(row)

def metrics_to_csv(test_target, predictions, test_round, test_X, model_name =None, predictors_num= None, pipeline = None, model_py = None, time = None):
    row = create_row(test_X, test_round, pipeline, model_name, predictors_num,model_py, time)
    initiate_csv(row)

    test_target = test_target.sort_index(axis = 0)
    predictions = predictions.sort_index(axis = 0)
    test_X = test_X.sort_index(axis = 0)

    if any(test_target.index != predictions.index) or any(test_X.index!=predictions.index):
        raise("the row indices of test_target and predictions are different")


    mv = test_X[[col_market, col_vehic]].drop_duplicates()
    mv.loc[mv.shape[0]] = ['all', 'all']
    row_all = []
    for i,mv_row in mv.iterrows():
        if mv_row['market'] != 'all' and mv_row['vehicle_line'] != 'all':
            temp = test_X.copy(deep =True)
            for col in mv_row.index.tolist():
                temp = temp[(temp[col] == mv_row[col])  ]
            idx = temp.index
        else:
            idx = test_X.index
        test_target_s = test_target.loc[idx]
        predictions_s = predictions.loc[idx]
        row['market'] = mv_row['market']
        row['vehicle_line'] = mv_row['vehicle_line']
        row['data_size'] = test_target_s.shape[0]
        metrics = cal_metrics(test_target_s, predictions_s, row['data_size'], row['predictors_num'])
        row.update(metrics)
        write_metrics(row)
        row_all.append({**row})

    return(row_all)


def overall_metrics_to_csv(row_alls, test_round, model_name =None, predictors_num= None, pipeline = None, model_py = None):
    cols = ['market', 'vehicle_line']
    unique = [{ k:v for  k,v in r.items() if k in cols}  for r in row_alls]
    unique = [dict(y) for y in set(tuple(x.items()) for x in unique)]

    for u in unique:
        row = create_row(test_X=None, test_round=test_round, pipeline=pipeline, model_name=model_name,
                         predictors_num=predictors_num, model_py = model_py, time = None)
        rows_temp = [r for r in row_alls if all([r[k] == v for k,v in u.items()])]
        test_start = min([r['test_start'] for r in rows_temp])
        test_end = max([r['test_end'] for r in rows_temp])
        time = sum([r['execution_time'] for r in rows_temp])/len([r['execution_time'] for r in rows_temp])
        row.update({'data_size': rows_temp[1]['data_size']})
        row.update({'test_start': test_start})
        row.update({'test_end': test_end})
        row.update({'execution_time': time})
        row.update(u)

        for i in metrics_list:
            metric = sum(r[i] for r in rows_temp)/len(rows_temp)
            row.update({i:metric})
        write_metrics(row)
