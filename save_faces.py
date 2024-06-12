import face_recognition
import cv2
import os
import redis
import numpy as np

# Initialize Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def save_face_and_encodings(name):
    video_capture = cv2.VideoCapture(0)
    image_count = 0
    user_folder = f'saved_faces/{name}'

    # Create a folder for the new face if it doesn't exist
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    while image_count < 4:
        ret, frame = video_capture.read()
        if not ret:
            continue

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all the faces in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Save the face image
            face_image = frame[top:bottom, left:right]
            cv2.imshow('Face', face_image)  # Display the face

            image_path = os.path.join(user_folder, f'{name}_{image_count}.jpg')
            cv2.imwrite(image_path, face_image)
            print(f'Image saved as {image_path}')

            # Save the encoding to Redis in a hash
            encoding_key = f'{name}_encodings'
            field_name = f'encoding_{image_count}'
            redis_client.hset(encoding_key, field_name, face_encoding.tobytes())
            print(f'Encoding saved in Redis hash with key {encoding_key} and field {field_name}')

            image_count += 1
            if image_count >= 4:
                break

        if image_count >= 4:
            break

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Example usage
user_name = input("Enter your name: ")
save_face_and_encodings(user_name)
