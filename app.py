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

# Directories for saving images
os.makedirs('known_faces', exist_ok=True)
os.makedirs('unknown_faces', exist_ok=True)
unknown_face_counter = 0  # Initialize counter for unknown face images
# unknown_face_encodings = []  # To track encodings of unknown faces

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

def load_unknown_faces(base_directory="unknown_faces"):
    unknown_faces = []
    unknown_identifiers = []

    # Ensure the directory exists
    if not os.path.exists(base_directory):
        print("Directory not found:", base_directory)
        return unknown_faces, unknown_identifiers

    # Walk through the directory tree
    for root, dirs, files in os.walk(base_directory):
        for file in files:
            if file.lower().endswith(".jpg"):  # Assumes all images are JPEGs
                path = os.path.join(root, file)
                image = face_recognition.load_image_file(path)
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    # Typically, there should be one face per image in this scenario
                    encoding = face_encodings[0]
                    unknown_faces.append(encoding)
                    unknown_identifiers.append(os.path.splitext(file)[0])  # Save the identifier without the file extension

    return unknown_faces, unknown_identifiers

# Usage
unknown_face_encodings, unknown_ids = load_unknown_faces()
print("unknown_face_name, ", unknown_ids)

# Dictionary to track the last seen faces and their last detected time
faces_in_previous_frame = {}

def is_new_unknown_face(face_encoding):
    if not unknown_face_encodings:
        return True
    # Calculate distances from new face encoding to all stored unknown face encodings
    distances = face_recognition.face_distance(unknown_face_encodings, face_encoding)
    # Consider face new if no stored encoding is close enough
    if all(dist > 0.6 for dist in distances):  # Threshold might need adjustment
        return True
    return False


def update_faces_in_frame(face_infos):
    global faces_in_previous_frame, unknown_face_counter
    now = datetime.now()
    current_seen = set(info['name'] for info in face_infos)

    new_faces = current_seen - set(faces_in_previous_frame.keys())
    print("new_faces: ", new_faces)

    for info in face_infos:
        name, encoding, frame, top, right, bottom, left = info.values()
        if name == "Unknown":
            if is_new_unknown_face(encoding):
                unknown_face_encodings.append(encoding)
                unknown_face_counter += 1
                identifier = f'unknown_{unknown_face_counter}'
                unknown_ids.append(identifier)
                image_path = os.path.join('unknown_faces', f'{identifier}.jpg')
                face_image = frame[top:bottom, left:right]
                cv2.imwrite(image_path, face_image)
                unknown_faces_collection.insert_one({"identifier": identifier, "timestamp": now, "event_type": "new detection"})
                faces_in_previous_frame[identifier] = {'last_seen': now, 'identifier': identifier}
            else:
                faces_in_previous_frame[name]['last_seen'] = now
        elif name not in faces_in_previous_frame:
            faces_in_previous_frame[name] = {'last_seen': now}
        else:
            faces_in_previous_frame[name]['last_seen'] = now
            faces_in_previous_frame[name]['new'] = False
    
    print("list(faces_in_previous_frame.keys(): ", list(faces_in_previous_frame.keys()))
    for face in list(faces_in_previous_frame.keys()):
        print("face: ", face)
        print("now - faces_in_previous_frame[face]['last_seen']", now - faces_in_previous_frame[face]['last_seen'])
        print("timedelta(seconds=30): ", timedelta(seconds=30))
        print("face not in current_seen: ", face not in current_seen)
        print("(now - faces_in_previous_frame[face]['last_seen'] > timedelta(seconds=30)):", (now - faces_in_previous_frame[face]['last_seen'] > timedelta(seconds=30)))
        if (now - faces_in_previous_frame[face]['last_seen'] > timedelta(seconds=30)) and face not in current_seen:
            del faces_in_previous_frame[face]
            
    for face in list(new_faces):
        print("Face =================================> ", face)
        print('''"unknown" in face.lower() and face != "Unknown":''', "unknown" in face.lower() and face != "Unknown")
        if face != "Unknown":
            if "unknown" in face.lower():
                image_path = os.path.join('unknown_faces', f'{face}.jpg')
                cv2.imwrite(image_path, frame)  # Save the image of the unknown face
                unknown_faces_collection.insert_one({"identifier": f'{face}', "timestamp": now, "event_type": "new detection"})
            else:
                known_faces_collection.insert_one({"name": face, "timestamp": now, "event_type": "new detection"})


def generate_frames():
    video_capture = cv2.VideoCapture(0)
    tolerance = 0.5
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rgb_small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_infos = []
        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=tolerance)
            unKnown_face_matches = face_recognition.compare_faces(unknown_face_encodings, encoding, tolerance=tolerance)
            # name = "Unknown" if not any(matches) and not any(unKnown_face_matches) else known_face_names[matches.index(True)]
            if any(matches):
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            elif any(unKnown_face_matches):
                first_match_index = unKnown_face_matches.index(True)
                name = unknown_ids[first_match_index]
            else:
                name = "Unknown"
            face_infos.append({
                'name': name, 'encoding': encoding, 'frame': frame, 
                'top': top*4, 'right': right*4, 'bottom': bottom*4, 'left': left*4
            })

        update_faces_in_frame(face_infos)

        for info in face_infos:
            cv2.rectangle(frame, (info['left'], info['top']), (info['right'], info['bottom']), (0, 0, 255), 2)
            cv2.rectangle(frame, (info['left'], info['bottom'] - 35), (info['right'], info['bottom']), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, info['name'], (info['left'] + 6, info['bottom'] - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
