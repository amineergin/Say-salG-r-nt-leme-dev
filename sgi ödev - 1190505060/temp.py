import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.VideoCapture(0)

while 1:
    ret,frame = img.read()
    son = cv2.Canny(frame, 20, 100)
    cv2.imshow("webcam", son)


cv2.waitKey(0)
cv2.destroyAllWindows()
