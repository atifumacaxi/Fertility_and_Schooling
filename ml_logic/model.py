from colorama import Fore, Style
import time
import pandas as pd
import numpy as np
from typing import Tuple

print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from keras import Model

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")

def preproc(data:pd.DataFrame) -> pd.DataFrame:
    '''
    Fines adjustments on dataset
    '''
    #Removing columns
    #data.drop(columns=['Unnamed: 0', 'Code'], inplace=True)
    data.drop(columns='Code', inplace=True)

    #Ordering by year and set it as index
    data.sort_values('Year', inplace=True)
    data.set_index('Year', inplace=True)

    return data


def create_X_y(data:pd.DataFrame) -> Tuple[list, list]:
    '''
    Given a countries dataset, this function returns
    two lists of dataframes, ie., lists containing one dataframe per country.
    '''
    data = preproc(data)
    countries = data.Country.unique().tolist()

    X = []
    y = []

    new_df = pd.DataFrame()

    for country in countries:
        new_df = data[data['Country']==country][['fertility', 'avg_years_of_schooling']]

        #This filter (34) is essential to keep our data normalized and standardized!
        if new_df.shape[0] == 34: #Considering only countries that has 34 samples (34 is the max number of samples)
            X.append(new_df.head(33))
            y.append(new_df['avg_years_of_schooling'].tail(1))
        else:
            pass

    #Transforming X to a numpy array
    X = np.array(X)
    #Transforming y to a numpy array and adding one dimension
    y = np.array(y)
    y = np.expand_dims(y.astype(np.float32), axis=-1)

    return X, y


def initialize_model(model: Model) -> Model:
    """
    Initialize the Neural Network
    """

    model = Sequential()
    model.add(SimpleRNN(units=20, activation='tanh'))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1, activation="linear"))

    print("✅ Model initialized")

    return model


def compile_model(model: Model) -> Model:
    """
    Compile the Neural Network
    """
    model.compile(loss='mse',
        optimizer='adam',
        metrics=['mae'])

    print("✅ Model compiled")

    return model


def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        patience=10,
        validation_data=None, # overrides validation_split
        validation_split=0.2
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=1000,
        callbacks=[es],
        verbose=1
    )
    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history
