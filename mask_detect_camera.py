import cv2
import tensorflow as tf
import numpy as np
import time
import copy

secBetween = 3
startingPoint = 100
calibComplete = False

maskChecks = 0
maskResult = 0
checkLimit = 5

hasMask = False
validResult = False

model_loaded = tf.keras.models.load_model('/home/kevin/Desktop/AI Project/maskface.model')

webcam = cv2.VideoCapture(0)

def isNewSubject(currentImage, calibratedImage):
    sensitivity = 10
    currentWeights = np.average(currentImage)
    calibWeights = np.average(calibratedImage)
    result = False

    if(currentWeights <= calibWeights - 12 or currentWeights >= calibWeights + 12):
        result = True
    else:
        result = False

    print("Current average: " + str(currentWeights))
    print("Calibration average: " + str(calibWeights) + "\n")

    return result

while(True):
    success, image = webcam.read()

    #if we haven't calibrated the camera yet, do it
    if(calibComplete == False):
        calibImage = copy.deepcopy(image)
        calibImage = calibImage[120:360, 210:430]
        calibImage = cv2.cvtColor(calibImage, cv2.COLOR_BGR2GRAY)
        calibComplete = True

    imageWithGuide = image
    imageModified = copy.deepcopy(image)
    imageBox = copy.deepcopy(image)
    imageBox = imageBox[120:360, 210:430]
    cv2.rectangle(imageWithGuide,(210, 120),(430, 360), (0,255,0),thickness=2)

    if(validResult):
        cv2.putText(imageWithGuide,str(hasMask),(50,500),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Live Feed", imageWithGuide)
    cv2.imshow("Calibrated image", calibImage)

    if(success):
        if(isNewSubject(imageBox,calibImage)):
            validResult = True
            #perform image preprocessing
            imageModified = cv2.cvtColor(imageModified, cv2.COLOR_BGR2GRAY)
            imageModified = imageModified[120:360, 210:430]
            imageModified = cv2.resize(imageModified,(150,150))
            cv2.imshow("Image to model", imageModified)
            imageModified = np.reshape(imageModified,(1,150,150,1))

            #pass to model, get result
            predictions = model_loaded.predict(imageModified) #include file to check out
            result = predictions[0]

            #Positive mask ID skews to 0
            # Negative on mask skews to 1
            if(result <= 0.5):
                #print("Mask\n")
                maskResult = maskResult + 1

            if(maskChecks == checkLimit):
                maskChecks = 0
                maskResult = (maskResult / checkLimit)

                if(maskResult >= 0.6):
                    print("Mask")
                    hasMask = True
                else:
                    print("No mask")
            else:
                maskChecks = maskChecks + 1
        else:
            validResult = False

    else:
        print("Camera did not open")
        break

    if(cv2.waitKey(1) == 27):
        break

    #time.sleep(1)

    

webcam.release()
cv2.destroyAllWindows() 