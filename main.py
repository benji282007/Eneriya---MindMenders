import cv2
import os
import face_recognition
import time
import speech_recognition as sr
import numpy as np
import threading
import json


MODEL = "hog"
MATCH_THRESHOLD = 0.5
PROCESS_EVERY_N_FRAMES = 5
UNKNOWN_CONFIRMATION_FRAMES = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "KnownFaces")
LEARNED_FACES_DIR = os.path.join(BASE_DIR, "LearnedFaces")
RELATIONSHIP_FILE = os.path.join(BASE_DIR, "relationships.json")

os.makedirs(LEARNED_FACES_DIR, exist_ok=True)


known_faces = []
known_names = []
relationships = {}

last_identity = "Unknown"
frame_count = 0
unknown_counter = 0
is_learning = False


def load_relationships():
    global relationships
    if os.path.exists(RELATIONSHIP_FILE):
        with open(RELATIONSHIP_FILE, "r") as f:
            relationships = json.load(f)

def save_relationships():
    with open(RELATIONSHIP_FILE, "w") as f:
        json.dump(relationships, f, indent=4)

def load_faces():
    if os.path.exists(KNOWN_FACES_DIR):
        for person in os.listdir(KNOWN_FACES_DIR):
            folder = os.path.join(KNOWN_FACES_DIR, person)
            if not os.path.isdir(folder):
                continue

            for file in os.listdir(folder):
                path = os.path.join(folder, file)
                image = face_recognition.load_image_file(path)
                enc = face_recognition.face_encodings(image)
                if enc:
                    known_faces.append(enc[0])
                    known_names.append(person)

    for file in os.listdir(LEARNED_FACES_DIR):
        path = os.path.join(LEARNED_FACES_DIR, file)
        image = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(image)
        if enc:
            name = file.split(".")[0]
            known_faces.append(enc[0])
            known_names.append(name)


def learning_process(face_crop):
    global is_learning, last_identity

    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:

            print("\n--- SAY YOUR NAME ---")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            audio = recognizer.listen(source, timeout=8, phrase_time_limit=5)
            name = recognizer.recognize_google(audio).strip()

            print("Heard Name:", name)

            time.sleep(1)

            print("\n--- SAY YOUR RELATIONSHIP ---")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)

            audio = recognizer.listen(source, timeout=8, phrase_time_limit=5)
            relationship = recognizer.recognize_google(audio).strip()

            print("Heard Relationship:", relationship)

    except Exception as e:
        print("Speech failed:", e)
        is_learning = False
        return

    save_path = os.path.join(LEARNED_FACES_DIR, f"{name}.jpg")
    cv2.imwrite(save_path, face_crop)

    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    enc = face_recognition.face_encodings(rgb)

    if enc:
        known_faces.append(enc[0])
        known_names.append(name)

    relationships[name] = relationship
    save_relationships()

    last_identity = name
    print(f"Stored: {name} - {relationship}")

    is_learning = False



def ask_relationship(identity):

    recognizer = sr.Recognizer()

    print(f"\nSay relationship for {identity}...")

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=6, phrase_time_limit=4)
            relationship = recognizer.recognize_google(audio).strip()

            relationships[identity] = relationship
            save_relationships()

            print(f"Stored relationship: {identity} - {relationship}")

    except Exception as e:
        print("Relationship capture failed:", e)


def recognize_face(rgb_small, frame):

    global last_identity, unknown_counter, is_learning

    if is_learning:
        return

    locations = face_recognition.face_locations(rgb_small, model=MODEL)
    encodings = face_recognition.face_encodings(rgb_small, locations)

    for encoding, location in zip(encodings, locations):

        if len(known_faces) == 0:
            return

        distances = face_recognition.face_distance(known_faces, encoding)
        best_index = np.argmin(distances)
        best_distance = distances[best_index]

        if best_distance < MATCH_THRESHOLD:

            identity = known_names[best_index]
            unknown_counter = 0

            if identity != last_identity:
                last_identity = identity

                if identity not in relationships:
                    threading.Thread(
                        target=ask_relationship,
                        args=(identity,),
                        daemon=True
                    ).start()

        else:
            unknown_counter += 1

            if unknown_counter >= UNKNOWN_CONFIRMATION_FRAMES:

                is_learning = True
                unknown_counter = 0

                top, right, bottom, left = location
                top *= 4; right *= 4; bottom *= 4; left *= 4

                face_crop = frame[top:bottom, left:right].copy()

                threading.Thread(
                    target=learning_process,
                    args=(face_crop,),
                    daemon=True
                ).start()


def main():
    global frame_count

    load_relationships()
    load_faces()

    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            recognize_face(rgb_small, frame)

        cv2.rectangle(frame, (0, 0), (900, 130), (0, 0, 0), -1)

        relationship = relationships.get(last_identity, "")
        display_text = f"{last_identity} - {relationship}"

        cv2.putText(frame,
                    display_text,
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3)

        current_time = time.strftime("%A | %I:%M %p")
        cv2.putText(frame,
                    current_time,
                    (20, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

        cv2.imshow("Memory Assistant", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
