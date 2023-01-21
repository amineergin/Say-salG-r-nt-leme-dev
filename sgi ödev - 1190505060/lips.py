import cv2
import numpy as np
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def creatBox(image,points,scale =5,filtered = False, clipped = True):
    if filtered:
        filter= np.zeros_like(image)
        filter = cv2.fillPoly(filter,[points],(255,255,255))
        img = cv2.bitwise_and(image,filter)

    if clipped:
        bbox = cv2.boundingRect(points)
        x,y,w,h = bbox
        imageClip = img[y:y+h,x:x+w]
        imageClip = cv2.resize(imageClip,(0,0),None,scale,scale)
        return imageClip

    else:
        return filter
    
while True:
        
    img_ = cv2.imread("sgiresim1.png")
    img_gray = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
    faces = detector(img_gray)
    
    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
        landmarks1 = predictor(img_gray,face)
        number = []

        for n in range(68):
            x = landmarks1.part(n).x
            y = landmarks1.part(n).y
            number.append([x,y])
            
        number = np.array(number)
        imgLips = creatBox(img_,number[48:61],3,filtered=True,clipped=False)
        img0 = img_.copy()
        imgLipsRed = cv2.fillPoly(img_, [number[48:61]],(0,0,220))
        imgRed = cv2.addWeighted(img_, 0.5, imgLipsRed, 0.3, 0.5)
        imgLipsGreen = cv2.fillPoly(img_, [number[48:61]],(74,128,77))
        imgGreen = cv2.addWeighted(img_, 0.5, imgLipsGreen, 0.3, 0.5)
        imgLipsYellow = cv2.fillPoly(img_, [number[48:61]],(110,250,250))
        imgYellow = cv2.addWeighted(img_, 0.5, imgLipsYellow, 0.3, 0.5)
        imgLipsPurple = cv2.fillPoly(img_, [number[48:61]],(139,101,139))
        imgPurple = cv2.addWeighted(img_, 0.5, imgLipsPurple, 0.3, 0.5)
    
        cv2.imshow('Original',img0)
        cv2.imshow('Red Lips',imgRed)
        cv2.imshow('Green Lips',imgGreen)
        cv2.imshow('Yellow Lips',imgYellow)
        cv2.imshow('Purple Lips',imgPurple)
        
        cv2.waitKey(0)    
    cv2.destroyAllWindows()