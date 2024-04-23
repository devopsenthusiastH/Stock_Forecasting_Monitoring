FROM python:3

WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit","run"]
CMD ["run.py","--server.port=8501","--server.address=0.0.0.0","--server.fileWatcherType","none","--browser.gatherUsageStats","false"]

