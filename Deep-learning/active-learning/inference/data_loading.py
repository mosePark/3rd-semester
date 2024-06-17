import pandas as pd

def load_data():
    data_pool = pd.read_csv('/home1/mose1103/koactive/JM-BERT/data/imbalance_data_pool_new.csv')
    start_data = pd.read_csv('/home1/mose1103/koactive/JM-BERT/data/start_data_new.csv')
    test_df = pd.read_csv('/home1/mose1103/koactive/JM-BERT/data/test_df_new.csv')

    # 레이블이 1에서 5사이로 되어 있을 때, 이를 0에서 4사이로 변환
    data_pool['score'] -= 1
    start_data['score'] -= 1
    test_df['score'] -= 1

    return data_pool, start_data, test_df
