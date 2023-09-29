FROM python:3.11.5

WORKDIR /app

COPY . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 5001

CMD ["python", "app.py"]
