FROM apache/airflow:slim-3.4.0-python3.12

# Java 17 is for Spark 3.5 (it documents Java 8/11/17; Java 21 support came with Spark 4.0).
# Debian 13 has no OpenJDK 17, so copy the Temurin JRE instead of using apt.
COPY --from=eclipse-temurin:17.0.21_9-jre-noble /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"

USER airflow
COPY base/requirements.txt /tmp/requirements.txt
# Two steps: the Airflow constraints need numpy 2 and pandas 3, but torch 2.3.0 needs numpy 1.x.
# Step 1 installs Airflow and its providers exactly as constrained (slim has no providers).
# Step 2 adds our pins; pip keeps Airflow's own packages (SQLAlchemy, psycopg2) because they already fit.
# Cache mount keeps wheels between builds without growing the image; long timeout is for slow networks.
RUN --mount=type=cache,target=/home/airflow/.cache/pip,uid=50000,gid=0 \
    PIP_DEFAULT_TIMEOUT=120 pip install "apache-airflow[celery,postgres,redis,standard]==${AIRFLOW_VERSION}" \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt"
RUN --mount=type=cache,target=/home/airflow/.cache/pip,uid=50000,gid=0 \
    PIP_DEFAULT_TIMEOUT=120 pip install "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements.txt
RUN pip check
