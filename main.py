from datetime import datetime as dt
import sys
sys.path.append('.\\proto_models')
import load_data as load
import preprocessing as prep
import metrics
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Which model.py would you like to run?')
    parser.add_argument('--models',nargs='+', help='specify which .py in proto_models you would like to run ')
    args = parser.parse_args()
    argsm = args.models
    if 'all' in argsm:
        files =  os.listdir('.\\proto_models')
        module_list =  [f.split('.py')[0] for f in files if f.endswith('.py')]
    else:
        module_list = argsm



    for mod in module_list:
        print(f'Running {mod}')
        model = __import__(mod)
        df = load.import_sales_data_from_bq()
        df = prep.clean_data(df)
        trains, tests = prep.train_test_split(df)

        row_alls = []

        for k,_ in trains.items():
            tic = dt.now()

            train_X = trains[k]['X']
            train_y = trains[k]['y']

            test_X = tests[k]['X']
            test_y = tests[k]['y']



            pipeline = model.construct_Pipline()

            pipeline.fit(train_X, train_y)
            predictions = pipeline.predict(test_X)

            toc = (dt.now() - tic).seconds

            rows = metrics.metrics_to_csv(test_y, predictions, test_round= k, test_X= test_X, pipeline = pipeline, model_py= mod, time = toc)
            row_alls.extend(rows)

            ###############################
        tr = 'all(average across test rounds)'
        metrics.overall_metrics_to_csv(row_alls = row_alls, test_round = tr,pipeline = pipeline, model_py=mod)
        print('csv of metrics has been produced.')
    print('End of program')

