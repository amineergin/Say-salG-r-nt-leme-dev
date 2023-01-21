import cv2
import os
import sys

img=cv2.imread("images/meryem_uzerli.jpg")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

fl=face_cascade.detectMultiScale(gray_img,1.3,7)

dog=cv2.imread("images/dog.jpg")


def put_dog_filter(dog, fc, x, y, w, h):
    face_width = w
    face_height = h

    dog = cv2.resize(dog, (int(face_width * 1.5), int(face_height * 1.95)))
    for i in range(int(face_height * 1.75)):
        for j in range(int(face_width * 1.5)):
            for k in range(3):
                if dog[i][j][k] < 235:
                    fc[y + i - int(0.375 * h) - 1][x + j - int(0.35 * w)][k] = dog[i][j][k]
    return fc

for (x, y, w, h) in fl:
    frame = put_dog_filter(dog, img, x, y, w, h)

cv2.imshow("image",frame)
cv2.waitKey(20000)& 0xff
cv2.destroyAllWindows()