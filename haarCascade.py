# -*- coding: utf-8 -*-

import cv2

face_cascade = cv2.CascadeClassifier('Dataset/haarcascade_frontalface_default.xml')
sideFace_cascade = cv2.CascadeClassifier('Dataset/haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('Dataset/haarcascade_eye.xml')
fullbody_cascade = cv2.CascadeClassifier('Dataset/haarcascade_fullbody.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 2)
    sideFaces = sideFace_cascade.detectMultiScale(gray, 1.2, 2)
    body = fullbody_cascade.detectMultiScale(gray, 1.2, 2)
    
    for (x,y,w,h) in body:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w] 
        
    for (x,y,w,h) in sideFaces:            
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)            
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
        
        
    for (x,y,w,h) in faces:            
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)            
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)

            
            
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
        