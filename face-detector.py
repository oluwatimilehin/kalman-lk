import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4,480)

face_cascade = cv.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml")


while True:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(len(faces))

    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey+ eh), (0, 255, 0), 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv.destroyAllWindows()