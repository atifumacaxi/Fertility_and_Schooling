FROM python:3.10.6-buster
EXPOSE 8080
WORKDIR /prod

COPY requirements_prod.txt requirements.txt

COPY fertility fertility
COPY setup.py setup.py
RUN pip install .

#COPY Makefile Makefile
#RUN make reset_local_files

CMD uvicorn fertility.api.fast:app --host 0.0.0.0 --port $PORT
