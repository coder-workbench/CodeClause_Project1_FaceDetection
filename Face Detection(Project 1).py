import cv2

cap = cv2.VideoCapture(0) 
faseCascade = cv2.CascadeClassifier("haar.xml")

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faseCascade.detectMultiScale(
        gray, 
        scaleFactor = 1.1, 
        minNeighbors = 10,
        minSize = (90,30),
    )

    # Face Detection Starts
    if len(faces) > 0:
        cv2.putText(frame, "Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (104, 204, 0), 2)
    else:
        cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (62, 64,64 ), 2)
    # Face Detection Ends
    
    #Face Detection Frame Designing
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 255),2)
    cv2.imshow('frame',frame)

    #Adding wait key to hold the frame
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
