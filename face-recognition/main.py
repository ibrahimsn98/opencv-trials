import cv2
import numpy as np
import os

dirs = os.listdir('Data')
faces = []
labels = []
names = []

recognizer = cv2.face.LBPHFaceRecognizer_create()
faceCascade = cv2.CascadeClassifier('Cascade/face.xml')

print('Starting to Prepare Data')

if len(dirs) == 0:
    print('No Data Available!')
else:

    for index, dir_name in enumerate(dirs):
        label = int(index)
        name = dir_name
        images = os.listdir('Data/' + dir_name)
        for image_name in images:
            image = cv2.imread('Data/' + dir_name + '/' + image_name)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            labels.append(label)
            faces.append(image)
            names.append(name)

    print('All Data Prepared, Face Recognizing is starting!')

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(labels))

    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(grayFrame, 1.2, 5)

        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x-20, y-20), (x+w+20, y+h+20), (0, 255, 0), 2)
            face = grayFrame[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face)

            if confidence < 65:
                faceName = names[labels.index(label)]
                cv2.putText(frame, faceName, (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                print('Detected: ' + faceName + ' Confidence: ' + str(confidence))
            else:
                cv2.putText(frame, 'Unknown', (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Face Recognize', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cam.release()