FROM python:3.10.6-buster

COPY requirements_prod.txt requirements.txt

COPY fertility fertility
COPY setup.py setup.py

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn fertility.api.fast:app --host 0.0.0.0 --port $PORT
