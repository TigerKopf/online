# Verwende ein schlankes Python-Basisimage
FROM python:3.10-slim-bullseye

# Setze das Arbeitsverzeichnis im Container
WORKDIR /app

# Kopiere das Python-Skript (test.py) in das Arbeitsverzeichnis des Containers
# Stelle sicher, dass test.py im selben Verzeichnis wie die Dockerfile liegt
COPY test.py .

# Installiere PyTorch explizit für CPU zuerst.
# Die Version 2.8.0 wurde in deinen Logs verwendet. Das "+cpu" und die --index-url
# stellen sicher, dass nur die CPU-Version installiert wird und keine CUDA-Abhängigkeiten gezogen werden.
RUN pip install --no-cache-dir \
    torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Installiere die restlichen Python-Abhängigkeiten.
# Diese sollten jetzt die bereits installierte CPU-Version von Torch erkennen und verwenden.
RUN pip install --no-cache-dir \
    transformers \
    scipy \
    numpy \
    tqdm

# Erstelle ein Verzeichnis für die Ausgabe, da das Skript dies erwartet
RUN mkdir -p /app/output

# Definiere den Befehl, der ausgeführt wird, wenn der Container gestartet wird
# Dies führt dein Python-Skript aus und startet die Musikgenerierung
CMD ["python", "test.py"]