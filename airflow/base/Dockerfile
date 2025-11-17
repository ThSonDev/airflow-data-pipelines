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
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    # Fix lỗi kafka.vendor.six.moves không tồn tại trong kafka-python 2.0.1
    sed -i "s|from kafka.vendor.six.moves import range|from six.moves import range|" \
        /home/airflow/.local/lib/python3.*/site-packages/kafka/codec.py || true

# ================================
# 🔧 Copy scripts và cấp quyền thực thi (chuẩn reproducible)
# ================================
USER root
COPY projects/absa_streaming/scripts /opt/airflow/projects/absa_streaming/scripts
RUN chmod +x /opt/airflow/projects/absa_streaming/scripts/*.sh
USER airflow

# Copy thư mục streamlit của project vào image
COPY projects/absa_streaming/streamlit /opt/airflow/projects/absa_streaming/streamlit

# Cài các thư viện cần thiết cho Streamlit
USER airflow
RUN pip install --no-cache-dir streamlit psycopg2-binary pandas
