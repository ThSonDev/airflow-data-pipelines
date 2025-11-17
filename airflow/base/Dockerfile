FROM apache/airflow:2.9.0

USER root
# Cài Java (cho Spark)
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
# Copy file requirements trong chính thư mục base

COPY base/requirements.txt /tmp/requirements.txt


USER airflow
RUN pip install --no-cache-dir -r /tmp/requirements.txt


USER root
COPY projects/absa_streaming/scripts /opt/airflow/projects/absa_streaming/scripts
RUN chmod +x /opt/airflow/projects/absa_streaming/scripts/*.sh


USER airflow
# Copy thư mục streamlit của project vào image
COPY projects/absa_streaming/streamlit /opt/airflow/projects/absa_streaming/streamlit
