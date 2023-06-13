import os
import glob
from fertility.params import *
from tensorflow import keras
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fertility.ml_logic.registry import loads_model

app = FastAPI()

app.state.model = loads_model()

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

@app.get("/predict")
def predict(value1:float, value2:float):

    model = app.state.model
    assert model is not None

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
    print (len([[[value1, value2]]]))
    #pred = model.predict([[[value1, value2]]])
    #print(pred)
    #return int(pred[0])
    return 'hello'
