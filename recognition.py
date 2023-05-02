import time

from flask import Flask, render_template, Response, stream_with_context, request
import cv2
import numpy as np
import face_recognition
import os
import math

app = Flask(__name__)

images = []
resource_name = []
resource_path = "faceResources"
resource_list = os.listdir(resource_path)

for cl in resource_list:
    curImg = cv2.imread(f'{resource_path}/{cl}')
    print(f'{resource_path}/{cl}')
    images.append(curImg)
    resource_name.append(os.path.splitext(cl)[0])


def accurancy_caculator(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        return (1.0 - face_distance) / (range * 2.0)
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))


def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list


encoded_source_list = find_encodings(images)
print("Server is ready to start...")

# video capture 
cap = cv2.VideoCapture(0)


def process_frame(img, encoded_source_list, class_names, scale_rate, detected_faces="", encoded_faces=""):
    scale_rate = float(scale_rate)
    scale_reverse = int(1 / scale_rate)
    scaled_img = cv2.resize(img, (0, 0), None, scale_rate, scale_rate)
    scaled_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2RGB)

    if detected_faces == "" or encoded_faces == "":
        detected_faces = face_recognition.face_locations(scaled_img, model="hog")
        encoded_faces = face_recognition.face_encodings(scaled_img, detected_faces)
    else:
        detected_faces = detected_faces
        encoded_faces = encoded_faces

    for encodeFace, faceLocation in zip(encoded_faces, detected_faces):
        matches = face_recognition.compare_faces(encoded_source_list, encodeFace)
        face_dist = face_recognition.face_distance(encoded_source_list, encodeFace)
        match_index = np.argmin(face_dist)

        if matches[match_index]:
            name = class_names[match_index].upper()
            accuracy = round(accurancy_caculator(face_dist[match_index]) * 100)
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * scale_reverse, x2 * scale_reverse, y2 * scale_reverse, x1 * scale_reverse
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name + " " + str(accuracy) + "%", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (255, 255, 255), 2)
        else:
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    return img, detected_faces, encoded_faces


def video_generator(encoded_source_list, class_names, scale_rate, frame):
    fps = 0
    prev_time = time.time()
    total_frames = 0
    current_frame = 0
    detected_faces = ""
    encoded_faces = ""

    while True:
        success, img = cap.read()

        if not success:
            break

        current_frame += 1
        curr_time = time.time()
        elapsed_time = curr_time - prev_time
        total_frames += 1
        if elapsed_time > 1:
            fps = total_frames / elapsed_time
            prev_time = curr_time
            total_frames = 0
        fps_text = f"FPS: {fps:.2f}"

        # if current_frame % int(frame) == 0:
        #     continue

        if current_frame % int(frame) == 0:
            processed_img,detected_faces,encoded_faces = process_frame(img, encoded_source_list, class_names, scale_rate,"","")
        else:
            processed_img,detected_faces,encoded_faces = process_frame(img, encoded_source_list, class_names, scale_rate,detected_faces,encoded_faces)
        cv2.putText(processed_img, f"Frame: {current_frame}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(processed_img, fps_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(processed_img, f"Scale rate: {scale_rate}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2)
        cv2.putText(processed_img, f"Frame skipped: {int(frame)-1}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255),
                    2)
        ret, buffer = cv2.imencode('.jpg', processed_img)
        img_bytes = buffer.tobytes()

        image_bytes = b''
        image_bytes += b'--frame\r\n'
        image_bytes += b'Content-Type: image/jpeg\r\n\r\n'
        image_bytes += img_bytes
        image_bytes += b'\r\n'

        yield image_bytes

    cap.release()
    cv2.destroyAllWindows()


# @app.route('/recognition_video')
# def recognition_video():
#     return Response(stream_with_context(video_generator(encoded_source_list, class_names)),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/recognition_video')
def recognition_video():
    scale = request.args.get('scale')
    frame = request.args.get('frame')
    return Response(stream_with_context(video_generator(encoded_source_list, resource_name, scale, frame)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


# @socketio.on('connect')
# def on_connect():
#     print('Client connected')
#
#
# @socketio.on('disconnect')
# def on_disconnect():
#     print('Client disconnected')
#
#
# @socketio.on('stream')
# def handle_stream():
#     print("Client requested video stream")
#     return emit('frame', gen_frames(), broadcast=True)
