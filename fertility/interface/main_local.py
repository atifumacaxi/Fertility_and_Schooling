#from fertility.ml_logic import clean_data
from fertility.ml_logic.preprocessing import preprocessing_features
from fertility.ml_logic.model import create_X_y, compile_model, initialize_model, train_model
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

def preprocess_and_train():

    #Loading schooling dataset
    df_schooling = pd.read_csv('fertility/data/mean-years-of-schooling-long-run.csv', sep=';')
    #Loading fertility dataset
    df_fertility = pd.read_csv('fertility/data/fertility_rate.csv')

    df = preprocessing_features(df_schooling, df_fertility)

    X, y = create_X_y(df)

    #Splits into train and test sets - Without shuffling, because is a time series.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)

    # Train a model on the training set, using `model.py`
    model = None
    model = initialize_model(model)
    model = compile_model(model)
    model, history = train_model(model, X_train, y_train)

    # Compute the validation metric (min val_mae) of the holdout set
    val_mae = np.min(history.history['val_mae'])

    print("✅ preprocess_and_train() done")

if __name__ == '__main__':
    try:
        preprocess_and_train()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
