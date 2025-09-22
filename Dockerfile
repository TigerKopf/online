# Verwende ein schlankes Python-Basisimage
FROM python:3.10-slim-bullseye

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere das Python-Skript (test.py) in das Arbeitsverzeichnis des Containers
# Stelle sicher, dass test.py im selben Verzeichnis wie die Dockerfile liegt
COPY test.py .

# Installiere alle notwendigen Python-Abhängigkeiten
# --no-cache-dir reduziert die Größe des Docker-Images
# Installiere torch für CPU (Standardverhalten von pip, wenn keine CUDA-Version angegeben ist)
RUN pip install --no-cache-dir \
    transformers \
    scipy \
    torch \
    numpy \
    tqdm

# Definiere den Befehl, der ausgeführt wird, wenn der Container gestartet wird
# Dies führt dein Python-Skript aus und startet die Musikgenerierung
CMD ["python", "test.py"]