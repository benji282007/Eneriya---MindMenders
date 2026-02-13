import cv2

# Camera initialization
cam = cv2.VideoCapture(1,cv2.CAP_DSHOW)

#main loop
while True:
    ret,frame = cam.read()
    frame = cv2.flip(frame,1)
    if ret:
        cv2.imshow("CAMERA",frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

#exit code
cam.release()
cv2.destroyAllWindows()