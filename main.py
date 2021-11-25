import cv2
import numpy as np
from mss import mss
from PIL import Image


def grab_screen(mss_object, bbox: dict[str, int]) -> np.ndarray:
    sct_img = mss_object.grab(monitor=bbox)

    img = Image.frombytes('RGB', (sct_img.size.width, sct_img.size.height), sct_img.rgb)
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return img_bgr


if __name__ == "__main__":

    mon = {"top": -888, "left": 610, "width": 605, "height": 132}
    sct = mss()

    while True:
        # monitor_1 = sct.monitors[1]
        cv2.imshow('test', grab_screen(sct, mon))
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
