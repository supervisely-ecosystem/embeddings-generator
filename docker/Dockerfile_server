FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 51000

RUN pip install clip-as-service

CMD ["python3", "-m", "clip_server"]