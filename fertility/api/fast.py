from fastapi import FastAPI
from fertility.interface.graphics import one_country

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'HTTP': '200 - OK'}

@app.get("/maps")
def maps(country):
    graphic = one_country(country)
    return 'Hi'
