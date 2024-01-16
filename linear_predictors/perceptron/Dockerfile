FROM python:3.10

RUN pip install -U scikit-learn
RUN pip install numpy
RUN python -m pip install -U matplotlib

WORKDIR .
COPY . .
CMD ["python", "./main.py"]
