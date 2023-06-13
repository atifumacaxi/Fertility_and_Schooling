import pandas as pd
from typing import Tuple

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    #Loading schooling dataset
    df_schooling = pd.read_csv('fertility/data/mean-years-of-schooling-long-run.csv', sep=';')
    #Loading fertility dataset
    df_fertility = pd.read_csv('fertility/data/fertility_rate.csv')

    return df_schooling, df_fertility
