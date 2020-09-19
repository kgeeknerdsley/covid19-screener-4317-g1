#Bottom line? The basic face detectors won't work- I think we need to generate a new Haar detector for masked and maskless faces

import cv2
import imutils
from imutils import face_utils
import dlib

#test the cv2 haar detector, then dlib and see which one better

cv2Haar = '/home/kevin/Desktop/AI Project/covidAI/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml'
imgpath = '/home/kevin/Desktop/AI Project/covidAI/Dataset images/mask5.png'

haardetector = cv2.CascadeClassifier(cv2Haar) #load detector with front face xml data
hogdetector = dlib.get_frontal_face_detector()

#process the image, size it down and change colorspace to gray
testimg = cv2.imread(imgpath)
imgCorr = imutils.resize(testimg,500)
imgCorr = cv2.cvtColor(imgCorr,cv2.COLOR_BGR2GRAY)
imgCorrDlib = imgCorr

#run the two detectors
face = haardetector.detectMultiScale(imgCorr,1.7,3)
faceDLIB = hogdetector(imgCorrDlib,1)

#if no face, it will just draw nothing
for(x,y,w,h) in face:
    imgCorr = cv2.rectangle(imgCorr,(x,y),(x+w,y+h), (255,0,0),2)

#catches dlib failing, if nothing in rectangle then just print failed
if(faceDLIB):
    (x2, y2, w2, h2) = face_utils.rect_to_bb(faceDLIB[0])
    cv2.rectangle(imgCorrDlib, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
else:
    print("Dlib got nothing")

print(face)
print(faceDLIB)

cv2.imshow("OpenCVs attempt", imgCorr)
cv2.imshow("Dlibs attempt",imgCorrDlib)
cv2.waitKey(0)
cv2.destroyAllWindows()


