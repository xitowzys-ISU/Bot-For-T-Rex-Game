import cv2
import numpy as np
from mss import mss
from PIL import Image

from skimage import io
from skimage.feature import match_template


class Object:
    def __init__(self, path):
        self.img = io.imread(path)
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.location = None


def grab_screen(mss_object, bbox: dict[str, int]) -> np.ndarray:
    sct_img = mss_object.grab(monitor=bbox)

    img = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return img_bgr


if __name__ == "__main__":

    player: Object = Object('./objects/dino_day.png')

    mon = {"top": -888, "left": 610, "width": 605, "height": 132}
    sct = mss()

    while True:
        # monitor_1 = sct.monitors[1]
        image = grab_screen(sct, mon)

        result = match_template(image, player.img)

        ij = np.unravel_index(np.argmax(result), result.shape)
        _, x, y = ij[::-1]
        hcoin, wcoin, _ = player.img.shape

        image = cv2.rectangle(image, (x, y), (x + wcoin, y + hcoin), (0, 255, 0), 2)
        cv2.imshow('test', image)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            # cv2.imwrite('dino_game.png', image)
            cv2.destroyAllWindows()
            break
