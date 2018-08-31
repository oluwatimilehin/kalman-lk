import numpy as np
import cv2
import get_points
from collections import namedtuple
import tracker as track


Rect = namedtuple('Rectangle', 'top_x top_y bottom_x bottom_y')


def run(source):
    cap = cv2.VideoCapture(source)

    print("Press 'p' to pause the video and start tracking")
    while True:
        ret, img = cap.read()
        if cv2.waitKey(10) == ord('p'):
            break
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
    cv2.destroyWindow('Image')

    points = get_points.run(img)
    rect = Rect(points[0][0], points[0][1], points[0][2], points[0][3])
    if not points:
        print("ERROR: No object to be tracked.")
        exit()

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)

    tracker = track.Tracker(rect)
    prev_image = None

    while True:
        # Read frame from device or file
        retval, img = cap.read()

        if not retval:
            print("Cannot capture frame device | CODE TERMINATING :(")
            cv2.destroyAllWindows()
            break

        if prev_image is not None:
            tracker.update(prev_image, img, rect)
            tracker.run()
            rect = tracker.rect


        cv2.rectangle(img, (rect.top_x, rect.top_y), (rect.bottom_x, rect.bottom_y), (0, 0, 255), 8)

        # print("Object tracked at [{}, {}] \r".format(pt1, pt2), )\

        prev_image = img

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        # Continue until the user presses ESC key
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    source = 'filmrole1.avi'
    run(source)
