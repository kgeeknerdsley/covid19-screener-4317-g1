import cv2
import tensorflow as tf
import numpy as np
import time
import copy

secBetween = 3
startingPoint = 100
calibComplete = False

model_loaded = tf.keras.models.load_model('/home/kevin/Desktop/AI Project/maskface.model')

webcam = cv2.VideoCapture(0)

while(True):
    success, image = webcam.read()

    if(calibComplete == False):
        calibImage = copy.deepcopy(image)
        calibImage = calibImage[120:360, 210:430]
        calibComplete = True

    imageWithGuide = image
    imageModified = copy.deepcopy(image)
    cv2.rectangle(imageWithGuide,(210, 120),(430, 360), (0,255,0),thickness=2)

    cv2.imshow("Live Feed", imageWithGuide)
    cv2.imshow("Calibrated image", calibImage)

    if(success):
        if(cv2.waitKey(1) == 32):
            imageModified = cv2.cvtColor(imageModified, cv2.COLOR_BGR2GRAY)
            imageModified = imageModified[120:360, 210:430]
            imageModified = cv2.resize(imageModified,(150,150))
            cv2.imshow("Image to model", imageModified)
            imageModified = np.reshape(imageModified,(1,150,150,1))

            #print(np.shape(imageModified))

            predictions = model_loaded.predict(imageModified) #include file to check out
            result = predictions[0]

            #Positive mask ID skews to 0
            # Negative on mask skews to 1
            if(result <= 0.5):
                print("Mask\n")
            else:
                print("No mask\n")

            print("Result: " , str(result))
        elif(cv2.waitKey(1) == 27):
            print("Exiting!")
            break

    else:
        print("Camera did not open")
        break

    

webcam.release()
cv2.destroyAllWindows() 