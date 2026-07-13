FROM python:3.12-slim

WORKDIR /app

# Dipendenze prima del codice sorgente per sfruttare la cache Docker.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag.py app.py ./
COPY documenti/ documenti/

ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV CHROMADB_HOST=chromadb
ENV CHROMADB_PORT=8000

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
