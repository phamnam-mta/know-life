FROM python:3.8-slim as base

USER root

ENV TZ=Asia/Ho_Chi_Minh
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    wget \
    python3 \
    #nginx \
    ipython3\
    build-essential\
    #python-dev\
    python3-dev\
    ca-certificates \
    python3-distutils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -U --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY data /app/data

ENV PYTHONPATH "${PYTHONPATH}:/app/src"

WORKDIR /app

USER 1001

CMD ["python", "./src/run.py"]
