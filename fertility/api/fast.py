import os
import glob
# from fertility.params import *
#from tensorflow import keras
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#from fertility.ml_logic.registry import loads_model

app = FastAPI()

# app.state.model = loads_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'HTTP': '200 - OK'}

@app.get("/maps")
def maps(country):
    #graphic = one_country(country)
    return 'Hi, testing maps'


@app.get("/past")
def past_data(country:str) -> dict:
    def set_past_df(data:pd.DataFrame,country:str) -> pd.DataFrame:

        df = data[data['Country']==country]

        df = df.drop(columns=['Country','Unnamed: 0', 'Code','avg_years_of_schooling'])
        df.set_index('Year', inplace=True)
        df.rename(columns={'fertility':country}, inplace=True)

        return df

    cur_path = os.path.dirname(__file__)
    treated_csv = os.path.join(cur_path, '..', 'data', 'treated.csv')
    past = pd.read_csv(treated_csv)
    df_past = set_past_df(past, country)

    return df_past


@app.get("/predict")
def predict(country:str) -> dict:
    def set_pred_df(data:pd.DataFrame,country:str) -> pd.DataFrame:
        df = data[data['Country']==country]
        df = df.drop(columns=['Country'])
        df.rename(columns={'yhat':country, 'ds': 'Year'}, inplace=True)
        df.set_index('Year', inplace=True)
        df = df.drop(columns=['Unnamed: 0'])

        return df

    cur_path = os.path.dirname(__file__)
    pred_file = os.path.join(cur_path, '..', 'data', 'predictions4countries.csv')

    pred = pd.read_csv(pred_file)

    df_pred = set_pred_df(pred, country)

    return df_pred

@app.get("/all")
def all_countries() -> dict:
    df_japan = pd.concat([pd.DataFrame(past_data('Japan')), pd.DataFrame(predict('Japan'))], axis=0)
    df_brazil = pd.concat([pd.DataFrame(past_data('Brazil')), pd.DataFrame(predict('Brazil'))], axis=0)
    df_yemen = pd.concat([pd.DataFrame(past_data('Yemen')), pd.DataFrame(predict('Yemen'))], axis=0)
    df_afg = pd.concat([pd.DataFrame(past_data('Afghanistan')), pd.DataFrame(predict('Afghanistan'))], axis=0)

    result = pd.concat([df_japan, df_brazil, df_yemen, df_afg], axis=1)
    return result


    #model = app.state.model
    #assert model is not None

    #X_pred = pd.DataFrame(locals(), index=[0])

    #Get the latest model version name by the timestamp on disk
    # local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    # local_model_paths = glob.glob(f"{local_model_directory}/*")

    # if not local_model_paths:
    #     return None

    # most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    #print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    # latest_model = keras.models.load_model(most_recent_model_path_on_disk)
    #print(X_pred)
    #print (len([[[value1, value2]]]))
    #pred = model.predict([[[value1, value2]]])
    #print(pred)
    #return int(pred[0])
