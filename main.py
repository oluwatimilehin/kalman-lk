import numpy as np
import cv2
import get_points

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

    if not points:
        print("ERROR: No object to be tracked.")
        exit()

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)


    while True:
        # Read frame from device or file
        retval, img = cap.read()
        if not retval:
            print("Cannot capture frame device | CODE TERMINATING :(")
            exit()
        # Update the tracker
        # tracker.update(img)
        # Get the position of the object, draw a
        # bounding box around it and display it.
        # rect = tracker.get_position()
        # pt1 = (int(rect.left()), int(rect.top()))
        # pt2 = (int(rect.right()), int(rect.bottom()))
        # cv2.rectangle(img, pt1, pt2, (255, 255, 255), 3)
        cv2.rectangle(img, (points[0][0], points[0][1]), (points[0][2], points[0][3]), (255, 255, 255), 3)
        # print("Object tracked at [{}, {}] \r".format(pt1, pt2), )
        # if dispLoc:
        #     loc = (int(rect.left()), int(rect.top() - 20))
        #     txt = "Object tracked at [{}, {}]".format(pt1, pt2)
        #     cv2.putText(img, txt, loc, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        # Continue until the user presses ESC key
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    source = 'three-four.mp4'
    run(source)
