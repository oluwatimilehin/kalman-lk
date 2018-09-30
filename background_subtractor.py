import cv2
from collections import namedtuple


Rect = namedtuple('Rectangle', 'top_x top_y bottom_x bottom_y')


class BackgroundSubtractor:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.contours = []

    def run(self, im):
        fgmask = self.fgbg.apply(im)  # [self.rect.top_y - rect_variance:self.rect.bottom_y + rect_variance,
        # self.rect.top_x - rect_variance:self.rect.bottom_x+rect_variance, 2])
        ret, thresh = cv2.threshold(fgmask, 127, 255, 0)
        im2, self.contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def get_suitable_rectangles(self, rect: Rect):
        rect_coordinates = []
        if len(self.contours) > 0:
            suitable_contours = [cnt for cnt in self.contours if len(cnt) > 0] # This is to reduce computation time
            for cnt in suitable_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 2 and h > 2:
                    bottom_x, bottom_y = x + w, y + h
                    if x >= rect.top_x - 50 and bottom_x <= rect.bottom_x + 50:
                        if y >= rect.top_y - 50 and bottom_y <= rect.bottom_y + 50:
                            rect_coordinates.append(Rect(x,y,bottom_x,bottom_y))

        return rect_coordinates

    @staticmethod
    def get_best_candidate(rect_coordinates: list, points):
        count_dict = {}
        for index, rect in enumerate(rect_coordinates):

            count = 0
            for point in points:
                if rect.top_x <= point[0] <= rect.bottom_x or rect.top_y <= point[1] <= rect.bottom_y:
                    count += 1

            count_dict[index] = count

        max_index = 0
        max_val = 0

        for k, v in count_dict.items():
            if v > max_val:
                max_val = v
                max_index = k

        best_rect = rect_coordinates[max_index]

        return best_rect
