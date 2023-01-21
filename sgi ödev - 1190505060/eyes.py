import cv2

image = cv2.imread("images/image_1.png")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = eye_cascade.detectMultiScale(image)

for box in faces:
    x,y,w,h = box
    x2,y2 = x + w, y + h
    cv2.rectangle(image, (x,y), (x2,y2), (0,255,0),2)
    
    image = cv2.resize(image, None, fx=1/3, fy=1/3, interpolation=cv2.INTER_AREA)
    
cv2.imshow("image", image)

cv2.waitKey()
cv2.destroyAllWindows()   