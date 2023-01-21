import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img=cv2.imread("images/scarlett.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
fl=face_cascade.detectMultiScale(gray,1.09,7)
ey=face_cascade.detectMultiScale(gray,1.09,7)

glass=cv2.imread('images/glass.png')

def put_glass(glass, fc, x, y, w, h):
    face_width = w
    face_height = h

    glass_width = face_width + 1
    glass_height = int(0.35 * face_height) + 1

    glass = cv2.resize(glass, (glass_width, glass_height))

    for i in range(glass_height):
        for j in range(glass_width):
            for k in range(3):
                if glass[i][j][k] < 235:
                    fc[y + i - int(-0.20 * face_height)][x + j][k] = glass[i][j][k]
    return fc

for (x, y, w, h) in ey:
    frame=put_glass(glass,img, x, y, w, h)
   
cv2.imshow('image',frame)
cv2.waitKey(8000)& 0xff
cv2.destroyAllWindows()
