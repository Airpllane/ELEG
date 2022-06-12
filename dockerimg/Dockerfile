# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN apt-get update -y

RUN apt-get install default-libmysqlclient-dev -y

RUN apt-get install build-essential -y

RUN apt-get install python-tk python3-tk tk-dev -y

RUN pip3 install -r requirements.txt

COPY . .

#CMD ["ls"]
CMD ["python3", "/app/gui2.py"]
#ENTRYPOINT ["python3"]
