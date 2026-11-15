FROM apache/airflow:2.9.0

USER root
# Java is for Spark; libgl1 and libglib2.0-0 are for OpenCV in vehicle_counting
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow
COPY base/requirements.txt /tmp/requirements.txt
# Pin apache-airflow so pip cannot change Airflow's own version or its dependencies
# Cache mount keeps wheels between builds without growing the image; long timeout is for slow networks
RUN --mount=type=cache,target=/home/airflow/.cache/pip,uid=50000,gid=0 \
    PIP_DEFAULT_TIMEOUT=120 pip install "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt
