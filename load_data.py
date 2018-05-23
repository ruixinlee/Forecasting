import pandas as pd
from google.cloud import bigquery as bq


bq_config = {'proj_id' : 'jlr-dl-cat',
             'dset_id' : '2017_Forecasting_Volume_NS_DEV',
             'tb_id' : 'DEV_Forecasting_NS_Master_VL'}

raw_path =".\\data\\input\\"


def init_bq_client(bq_config):
    return(bq.Client(bq_config['proj_id']))

def get_data(q_job): #q_job = client.query(q)
    q_rows = q_job.result()
    col_name = [f.name for f in q_rows.schema]

    data = []
    for r in q_rows:
        data.append({f:r[i] for i,f in enumerate(col_name)})
    df = pd.DataFrame.from_records(data)
    return(df)

def load_data_from_bq(q, bq_config=bq_config):
    client = init_bq_client(bq_config)
    q_job = client.query(q)
    df = get_data(q_job)
    return(df)

def bq_import_markets_models(bq_config=bq_config):
    q = f"SELECT distinct level_3_reporting, model FROM `{bq_config['proj_id']}.{bq_config['dset_id']}.{bq_config['tb_id']}`"
    df = load_data_from_bq(q,bq_config)
    return(df)


def import_sales_permm_from_bq(bq_config=bq_config):
    df_model = bq_import_markets_models(bq_config)

    for i, row in df_model.iterrows():

        q = f"""SELECT * FROM `{bq_config['proj_id']}.{bq_config['dset_id']}.{bq_config['tb_id']}`
                where level_3_reporting ='{row['level_3_reporting']}'
                and model = '{row['model']}'"""

        df = load_data_from_bq(q,bq_config)
        df.to_csv(raw_path + f"{row['level_3_reporting']}-{row['model']}.csv")
    return(df)

def import_sales_data_from_bq(bq_config=bq_config):

    q = f"""SELECT * FROM `{bq_config['proj_id']}.{bq_config['dset_id']}.{bq_config['tb_id']}` where target_month  <= '2018-04-01' """

    df = load_data_from_bq(q,bq_config)
    return(df)


if __name__ == '__main__':
    import_sales_data_from_bq()