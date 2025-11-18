import cv2
import time
import json
import base64
import argparse
from confluent_kafka import Producer

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}]')

def produce_frames(video_path, topic, kafka_broker='kafka:9092'):
    conf = {'bootstrap.servers': kafka_broker, 'message.max.bytes': 10485760} # 10MB limit
    producer = Producer(conf)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps) # Extract 1 frame every second
    
    frame_id = 0
    processed_count = 0
    
    print(f"Starting Producer for {video_path} -> {topic} at 1 FPS")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only process 1 frame per second
        if frame_id % frame_interval == 0:
            video_second = frame_id // int(fps)
            
            # Resize to reduce Kafka payload (e.g., 640px width)
            height, width = frame.shape[:2]
            new_width = 640
            new_height = int(height * (new_width / width))
            frame_resized = cv2.resize(frame, (new_width, new_height))

            # Encode frame to JPEG -> Base64
            _, buffer = cv2.imencode('.jpg', frame_resized)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            message = {
                'frame_id': frame_id,
                'video_second': video_second,
                'image_b64': jpg_as_text,
                'source_video': topic.replace('_topic', '') # e.g., video1
            }

            producer.produce(
                topic, 
                json.dumps(message).encode('utf-8'), 
                callback=delivery_report
            )
            producer.poll(0)
            processed_count += 1
            
            # Simulate real-time streaming (1 sec delay)
            time.sleep(1)

        frame_id += 1

    cap.release()
    producer.flush()
    print(f"Finished. Sent {processed_count} frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--topic', type=str, required=True)
    parser.add_argument('--broker', type=str, default='kafka:9092')
    args = parser.parse_args()

    produce_frames(args.video_path, args.topic, args.broker)