from fastapi import FastAPI

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'HTTP': '200 - OK'}

@app.get("/maps")
def maps():
    return 'Hi'
