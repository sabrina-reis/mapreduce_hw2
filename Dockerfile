FROM python:latest

WORKDIR /app

RUN pip install --no-cache-dir rpyc

COPY . /app

CMD ["python3"]
