FROM python:3.7-slim

LABEL DEVELOPER="Abdullah Al Zubaer"

WORKDIR /app

COPY covtype.csv main.py requirements.txt .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["python", "main.py"]
