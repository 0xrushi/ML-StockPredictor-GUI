FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN apt update && apt install -y build-essential git python3-pip

RUN git clone https://github.com/NSEDownload/NSEDownload && pip install NSEDownload/dist/*

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

EXPOSE 8501