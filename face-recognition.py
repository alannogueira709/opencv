## you can access the tutorial for this code in here: https://www.youtube.com/watch?v=dY29JzuMJJU

import cv2
import face_recognition

known_face_encodigns = []
known_face_names = []


known_person1_image = face_recognition.load_image_file("dataset/alan.jpg")
known_person2_image = face_recognition.load_image_file("dataset/elber.jpeg")
known_person3_image = face_recognition.load_image_file("dataset/gustavo.jpg")
known_person4_image = face_recognition.load_image_file("dataset/rinaldo.jpg")
known_person5_image = face_recognition.load_image_file("dataset/guizuxo.jpg")


known_person1_encoding = face_recognition.face_encodings(known_person1_image)[0]
known_person2_encoding = face_recognition.face_encodings(known_person2_image)[0]
known_person3_encoding = face_recognition.face_encodings(known_person3_image)[0]
known_person4_encoding = face_recognition.face_encodings(known_person4_image)[0]
known_person5_encoding = face_recognition.face_encodings(known_person5_image)[0]


known_face_encodigns.append(known_person1_encoding)
known_face_encodigns.append(known_person2_encoding)
known_face_encodigns.append(known_person3_encoding)
known_face_encodigns.append(known_person4_encoding)
known_face_encodigns.append(known_person5_encoding)

known_face_names.append("Alan")
known_face_names.append("Elber")
known_face_names.append("Gustavo")
known_face_names.append("Rinaldo")
known_face_names.append("Guilherme")

camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodigns , face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top -10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF ==  ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()