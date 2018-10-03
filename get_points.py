import cv2


def run(img):
    img_copy = img.copy()
    im_disp = img.copy()

    window_name = "Select objects to be tracked here."
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img_copy)

    # Continue until the user presses ESC key
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.destroyAllWindows()
        return

    run.mouse_down = False
    points_1 = []
    points_2 = []
    rect = []

    def callback(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points_1.append((x, y))
            run.mouse_down = True
        elif event == cv2.EVENT_LBUTTONUP and run.mouse_down:
            run.mouse_down = False
            points_2.append((x, y))
            print("Object selected at [{}, {}]".format(points_1[0], (x, y)))
        elif event == cv2.EVENT_MOUSEMOVE and run.mouse_down:
            img_copy = img.copy()
            cv2.rectangle(img_copy, points_1[0], (x,y), (255, 255, 255), 3)
            cv2.imshow(window_name, img_copy)

    print("Press and release mouse around the object to be tracked.")
    cv2.setMouseCallback(window_name, callback)

    while True:
        # Draw the rectangular boxes on the image
        window_name_2 = "Objects to be tracked."
        for pt1, pt2 in zip(points_1, points_2):
            cv2.rectangle(im_disp, pt1, pt2, (255, 255, 255), 3)
        # Display the cropped images
        cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name_2, im_disp)
        key = cv2.waitKey(30)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            return

        if k == ord('p'):
            # Press key `s` to return the selected points
            cv2.destroyAllWindows()

            # tl - top left, #br- bottom right
            # tl + br is not an addition, but rather a concatenation to form a list.
            point = [(tl + br) for tl, br in zip(points_1, points_2)]
            # corrected_point = check_point(point)
            return point
        elif k == ord('q'):
            # Press key `q` to quit the program
            print("Quitting without saving.")
            cv2.destroyAllWindows()
            # Continue until the user presses ESC key
        elif k == ord('d'):
            # Press ket `d` to delete the last rectangular region
            if run.mouse_down == False and points_1:
                print("Object deleted at  [{}, {}]".format(points_1[-1], points_2[-1]))
                points_1.pop()
                points_2.pop()
                im_disp = img.copy()
            else:
                print("No object to delete.")


