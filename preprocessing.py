import load_data as load
import pandas as pd

def clean_data(df):

    date_col = ['target_month', 'IHS_time_of_prediction']

    for col in date_col:
        df[col] = pd.to_datetime(df[col])


    return(df)

def clean_data_VL_day(df):

    date_col = ['target_month', 'most_recent_prediction_made_date']

    for col in date_col:
        df[col] = pd.to_datetime(df[col])


    return(df)


def train_test_split(df):
    target_col = 'actual_sales'
    calendar_col = 'target_month'
    test_length = 12 # 1 year
    train_length = 24 # 2 years
    test_periods = 8
    test_start = df[calendar_col].max() - pd.DateOffset(months=test_length + test_periods - 2)
    test_starts = [test_start + pd.DateOffset(months=i) for i in range(test_periods)]
    test_ends = [d + pd.DateOffset(months = test_length) for d in test_starts]

    train_starts = [d - pd.DateOffset(months = train_length) for d in test_starts]

    trains = {}
    tests = {}

    for i,_ in enumerate(test_starts):
        df_train = df[(df[calendar_col] >= train_starts[i]) & (df[calendar_col]< test_starts[i])]
        df_test = df[(df[calendar_col] >= test_starts[i]) & (df[calendar_col] < test_ends[i])]

        trains[i] = {'X': df_train.drop(target_col, axis = 1), 'y':df_train[target_col]}
        tests[i] = {'X': df_test.drop(target_col, axis = 1), 'y':df_test[target_col]}

    return(trains, tests)