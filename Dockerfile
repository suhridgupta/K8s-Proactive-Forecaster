FROM python:3.7
ENV PYTHONUNBUFFERED=1
RUN pip install numpy
RUN pip install pandas
RUN pip install tensorflow
RUN pip install scipy
RUN pip install Flask
RUN pip install prometheus-client
COPY proactive-forecaster.py /proactive-forecaster.py
CMD flask --app /proactive-forecaster run --port 8082
