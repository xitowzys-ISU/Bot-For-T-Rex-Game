import cv2
import pyautogui
import numpy as np
from mss import mss

mon = {"top": 210, "left": 60, "width": 605, "height": 120}
pyautogui.PAUSE = 0
right, left = 140, 72

sct = mss()

while True:
    img = cv2.cvtColor(np.array(sct.grab(mon)), cv2.COLOR_BGRA2GRAY)
    img = cv2.resize(img, (500, 120))

    if 83 in img[88, left:right]:
        pyautogui.keyUp('down')
        pyautogui.keyDown('up')

    else:
        pyautogui.keyUp('up')
        pyautogui.keyDown('down')

    cv2.imshow("Dino BOT", img[88:100, left:right])

    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
