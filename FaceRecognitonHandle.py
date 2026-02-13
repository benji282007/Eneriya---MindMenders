import cv2
import os
import face_recognition 


# Camera initialization
cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)


MODEL = "hog"

knownFacesDir = "KnownFaces"
sampleImagePath = "UnknownFaces/" 
Tolerence = 0.4

knownFaces = []
knownNames = []

unknownFaces = []

def LoadFaces():
    for name in os.listdir(f"{knownFacesDir}"):
        for filename in os.listdir(f"{knownFacesDir}/{name}"):
            image = face_recognition.load_image_file(f"{knownFacesDir}/{name}/{filename}")
            encodings = face_recognition.face_encodings(image,model=MODEL)
            if(len(encodings)>0):
                knownFaces.append(encodings[0])
                knownNames.append(name)
                print(f"loaded image:{knownFacesDir}/{name}/{filename}")
            else:
                 print(f"failed to recognize:{knownFacesDir}/{name}/{filename}")

def RecognizeFace(img,frame):
    locations = face_recognition.face_locations(img,model=MODEL)
    encodings = face_recognition.face_encodings(img,model=MODEL)

    for encoding,location in zip(encodings,locations):
       
       detection = face_recognition.compare_faces(knownFaces,encoding,Tolerence) 
       identity = "unknown"
       top_left = (location[3],location[0])
       bottom_right = (location[1],location[2])
       color = [0,255,0]
       if True in detection :
           identity = knownNames[detection.index(True)]
           print(f"Match Found {identity}")
           cv2.rectangle(frame,top_left,bottom_right,color,2)
       else:
            print("Unknown")
            cv2.rectangle(frame,top_left,bottom_right,(0,0,255),2)


LoadFaces()
#main loop
while True:
    ret,frame = cam.read()
    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if ret:

        RecognizeFace(rgb_frame,frame)
        cv2.imshow("CAMERA",frame)
    if cv2.waitKey(1) == ord('q'):
        break

#exit code
cam.release()
cv2.destroyAllWindows()