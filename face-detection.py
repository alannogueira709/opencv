## tutorial available in: https://www.youtube.com/watch?v=5cg_yggtkso

import pathlib
import cv2


cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"

clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for(x,y,width, height ) in faces:
        cv2.rectangle(frame , (x,y), (x+width, y+height), (36,255,12), 2)
        ##cv2.putText(image, 'Rosto detectado', (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()