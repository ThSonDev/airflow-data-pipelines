#!/bin/bash
# ==========================================
# Script: run_producer.sh
# Chức năng:
#   - Khởi động Kafka Producer gửi dữ liệu vào topic "absa-reviews"
#   - Nếu gặp lỗi sẽ trả về exit code != 0 để Airflow retry
# ==========================================

set -e  # Dừng script ngay nếu có lỗi

echo "[Producer] Starting Kafka Producer..."
python /opt/airflow/projects/absa_streaming/scripts/producer.py

status=$?
if [ $status -eq 0 ]; then
  echo "[Producer] Completed successfully."
else
  echo "[Producer] Failed with exit code $status."
fi

exit $status
