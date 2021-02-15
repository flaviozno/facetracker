import cv2 
from random import randrange #isso é só frescura mesmo

#database
trained_face_data = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    
    #transformar a imagem em cinza
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detectar rosto
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #criar retangulo envolta do rosto
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(255),randrange(255),randrange(255)), 2)

    cv2.imshow('Quem eh mono MASTER?????', frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break


print("Code Completed")