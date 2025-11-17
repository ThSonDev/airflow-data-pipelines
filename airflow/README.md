Big Data Streaming ABSA System (Airflow – Spark – Kafka – PostgreSQL – Streamlit)



Hệ thống xử lý dữ liệu thời gian thực (real-time streaming) cho bài toán phân tích cảm xúc theo chủ đề (Aspect-Based Sentiment Analysis – ABSA). Pipeline sử dụng Kafka để truyền dữ liệu, Spark Structured Streaming để xử lý, Airflow để điều phối, PostgreSQL làm nơi lưu kết quả, và Streamlit để hiển thị dashboard real-time. Sinh viên cần tự build môi trường Docker và chạy hệ thống để quan sát toàn bộ vòng đời của pipeline.



Hướng dẫn chạy hệ thống:



1\. Giải nén file project (airflow.zip) vào bất kỳ vị trí nào trên máy (ví dụ D:\\BigData\\airflow\\).



2\. Mở PowerShell và chuyển vào thư mục project:

cd D:\\BigData\\airflow



3\. Build lại toàn bộ image (làm lần đầu):

docker compose build --no-cache



4\. Khởi động toàn bộ hệ thống:

docker compose up -d



5\. Mở web app để kiểm tra:

\- Airflow Web UI: truy cập http://localhost:8080

&nbsp; Đăng nhập:

&nbsp; username: airflow

&nbsp; password: airflow

&nbsp; Trong Airflow, bật DAG có tên absa\_streaming\_lifecycle\_demo, sau đó trigger thủ công (Run) để khởi động pipeline streaming gồm producer, consumer và các tác vụ giám sát.



\- Streamlit Dashboard: truy cập http://localhost:8501

&nbsp; Ứng dụng hiển thị kết quả phân tích cảm xúc theo thời gian, theo chủ đề (aspect), và thống kê cảm xúc tổng hợp trong cơ sở dữ liệu. Dữ liệu sẽ tự hiển thị sau từ PostgreSQL.



6\. Dừng hệ thống:

docker compose down



Lưu ý:

\- Không đổi tên hoặc di chuyển file docker-compose.yaml ra khỏi thư mục gốc.

\- Không cần tải hoặc import file .tar image. Hệ thống sẽ tự build từ Dockerfile.

\- Lần chạy đầu tiên có thể mất 10-20 phút do Docker tải thư viện.

\- Sau khi khởi động thành công, các container sẽ được lưu trong Docker Desktop.

\- Những lần sau, chỉ cần chạy:

docker compose up -d



Cấu trúc thư mục chính:

C:\\airflow

│

├── base\\ ← Dockerfile + requirements.txt cho image cơ sở

├── dags\\ ← các file DAG của Airflow

├── models\\ ← mô hình ABSA (.pt) hoặc file dummy để chạy thử

├── projects\\absa\_streaming\\ ← code xử lý producer, consumer, streamlit app

├── logs\\ ← log runtime của Airflow

├── docker-compose.yaml

└── README.md



Học phần: SE363 – Phát triển ứng dụng trên nền tảng dữ liệu lớn  

Ngành Kỹ thuật phần mềm – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM  

Thực hiện bởi: HopDT – Faculty of Software Engineering, University of Information Technology (FSE-UIT)

