import numpy as np
import cv2
cap = cv2.VideoCapture('dataset/filmrole2.avi')
fgbg = cv2.createBackgroundSubtractorMOG2() # cv2.bgsegm.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()
    # frame = frame[1:600, 3:400,2]
    fgmask = fgbg.apply(frame[1:1000, 3:600, 2])
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # cv2.imshow('frame', fgmask)
    new_ret, thresh = cv2.threshold(fgmask, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tot_w = []
    tot_h = []
    if len(contours) > 1:
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            tot_w.append(w)
            tot_h.append(h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

print(tot_w)
print(tot_h)