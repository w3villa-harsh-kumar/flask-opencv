from flask import Flask, render_template, Response
import cv2
import face_recognition
import pymongo
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# Setup MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27018/")
db = client["attendance"]
known_faces_collection = db["known_faces"]
unknown_faces_collection = db["unknown_faces"]

# Load known faces
def load_known_faces(base_directory="known_faces"):
    known_faces = []
    known_names = []
    # Walk through the directory tree
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.endswith(".jpg"):  # Assumes all images are JPEGs
                path = os.path.join(root, file)
                image = face_recognition.load_image_file(path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    encoding = face_encodings[0]
                    known_faces.append(encoding)
                    # Assuming the subfolder's name is the person's name
                    known_names.append(os.path.basename(root))
    return known_faces, known_names

known_face_encodings, known_face_names = load_known_faces()

# Dictionary to track the last seen faces and their last detected time
faces_in_previous_frame = {}

def update_faces_in_frame(current_faces):
    global faces_in_previous_frame
    now = datetime.now()
    current_seen = set(current_faces)
    previously_seen = set(faces_in_previous_frame.keys())

    new_faces = current_seen - previously_seen  # Determine new faces
    print("CUrrent seen", current_seen, current_faces)

    # Update last seen time for still present faces and add new faces
    for face in current_seen:
        if face not in previously_seen:
            faces_in_previous_frame[face] = {'last_seen': now, 'new': True}
        else:
            faces_in_previous_frame[face]['last_seen'] = now
            faces_in_previous_frame[face]['new'] = False  # Mark as not new since it's seen again

    # Remove faces not seen anymore or that have timed out
    for face in list(previously_seen):
        print("now - faces_in_previous_frame[face]['last_seen']:", now - faces_in_previous_frame[face]['last_seen'])
        print("timedelta(seconds=30): ",timedelta(seconds=30))
        print("face not in current_seen: ",face not in current_seen)
        print(now - faces_in_previous_frame[face]['last_seen'] > timedelta(seconds=30))
        if (now - faces_in_previous_frame[face]['last_seen'] > timedelta(seconds=30)) and face not in current_seen:
            del faces_in_previous_frame[face]

    # Add new faces to the database
    for face in new_faces:
        print(f"Inserted New face into DB: {face}")
        known_faces_collection.insert_one({"name": face, "timestamp": now, "event_type": "new detection"})


def generate_frames():
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)
        update_faces_in_frame(face_names)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
