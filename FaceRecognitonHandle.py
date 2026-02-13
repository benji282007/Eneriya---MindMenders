import time
import cv2
import os
import face_recognition
import delete_duplicate_images

# --- Configuration ---
MODEL = "hog" # Use "cnn" if you have a GPU
KNOWN_FACES_DIR = "KnownFaces"
TOLERANCE = 0.4
FRAME_SKIP = 3 # Only process every 3rd frame for speed

# --- State ---
known_faces = []
known_names = []
unknown_counter = 0

# Camera initialization
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def load_all_faces():
    """Initial load of the entire database."""
    print("Initializing database...")
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)
        
    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.isdir(person_dir):
            load_specific_person(name)

def load_specific_person(name):
    """Encodes only the images for a specific person and adds to memory."""
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    for filename in os.listdir(person_dir):
        path = os.path.join(person_dir, filename)
        try:
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image, model=MODEL)
            if encodings:
                known_faces.append(encodings[0])
                known_names.append(name)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    print(f"Loaded identity: {name}")

def add_new_face_sequence():
    """Captures a new person, saves frames, and updates encodings incrementally."""
    global unknown_counter
    
    # Generate a unique name/ID for the new person
    new_id = f"User_{int(time.time())}"
    new_path = os.path.join(KNOWN_FACES_DIR, new_id)
    os.makedirs(new_path)

    # 1. Capture Video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('temp.mp4', fourcc, 20.0, (640, 480))
    
    print(f"Recording new face: {new_id}...")
    start_time = time.time()
    while (time.time() - start_time) < 5: # Reduced to 5 seconds for efficiency
        ret, frame = cam.read()
        if ret:
            out.write(frame)
            cv2.putText(frame, "RECORDING NEW FACE...", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("CAMERA", frame)
            cv2.waitKey(1)
    
    out.release()
    
    # 2. Extract & Clean
    extract_frames_to_folder('temp.mp4', new_path)
    os.remove('temp.mp4')
    
    # Optional: Deduplicate (using your existing module)
    try:
        delete_duplicate_images.process_images(new_path, threshold=4)
    except:
        pass

    # 3. INCREMENTAL LOAD: Only load the new person
    load_specific_person(new_id)
    unknown_counter = 0

def extract_frames_to_folder(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % 5 == 0: # Save every 5th frame to avoid massive redundancy
            cv2.imwrite(os.path.join(output_folder, f"face_{count}.jpg"), frame)
        count += 1
    cap.release()

def recognize_logic(rgb_frame, original_frame):
    global unknown_counter
    
    locations = face_recognition.face_locations(rgb_frame, model=MODEL)
    encodings = face_recognition.face_encodings(rgb_frame, locations)

    for encoding, location in zip(encodings, locations):
        matches = face_recognition.compare_faces(known_faces, encoding, TOLERANCE)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
            unknown_counter = 0
            color = (0, 255, 0)
        else:
            unknown_counter += 1
            color = (0, 0, 255)

        # Draw box
        top, right, bottom, left = location
        cv2.rectangle(original_frame, (left, top), (right, bottom), color, 2)
        cv2.putText(original_frame, name, (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if unknown_counter > 15: # Threshold for triggering new face capture
        add_new_face_sequence()

def main():
    load_all_faces()
    frame_count = 0
    
    while True:
        ret, frame = cam.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        # Skip frames to boost FPS
        if frame_count % FRAME_SKIP == 0:
            rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            recognize_logic(rgb_small, frame)
            
        cv2.imshow("CAMERA", frame)
        frame_count += 1
        
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()