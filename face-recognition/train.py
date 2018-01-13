import cv2
import os

faceCascade = cv2.CascadeClassifier('Cascade/face.xml')
counter = 1
faceData = []
faceData.append(0)

print('Please enter your name!')

subjectName = raw_input('Your name: ')

print('We need your 5 face image. Please press s for shot!')

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = faceCascade.detectMultiScale(grayFrame, 1.2, 5)

    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x-50, y-50), (x+w+50, y+h+50), (225, 0, 0), 2)
        faceData[0] = grayFrame[y:y+h, x:x+w]

    cv2.imshow('Training', frame)

    if cv2.waitKey(10) & 0xFF == ord('s'):
        if not os.path.exists('Data/' + subjectName):
            os.makedirs('Data/' + subjectName)
        cv2.imwrite('Data/' + subjectName + '/' + str(counter) + '.jpg', faceData[0])
        print(str(counter) + ' image trained!')
        counter += 1
        if counter == 6:
            print('Training completed!')
            break

cv2.destroyAllWindows()





