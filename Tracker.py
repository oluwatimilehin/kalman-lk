from main import Rect
import lucas_kanade

def measure(rect: Rect, im1, im2):
    lucas_kanade.run(rect, im1, im2)

def run(rect: Rect, im1, im2):


    new_rect = Rect(rect.top_x + max_v, rect.top_y + max_u, rect.bottom_x + max_v, rect.bottom_y + max_u)

    return new_rect
