#imports for temperature sensor

#imports for face detector
import tensorflow as tf
import cv2
import numpy as np
import time
import copy

#imports for symptom voice code

#setup for loading models, etc

#this filepath will need to be changed on the Pi, the model is about 1.2GB in size
#the Pi might struggle to do this inference in realtime, tensorflow lite may be answer
#or we just demo on our own machines
model_loaded = tf.keras.models.load_model('/home/kevin/Desktop/AI Project/maskface.model')

#checkpoint booleans
passedTemp = False
passedMask = False
passedSymptoms = False

#any functions we may need
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


#TEMPERATURE CODE HERE
#
#
#
#

#FACE CODE HERE
secBetween = 3
startingPoint = 100
calibComplete = False

sampleCount = 0
sampleVals = 0

webcam = cv2.VideoCapture(0)

while(True): #will probably need to break this differently, is currently infinite
    success, image = webcam.read()

    #if we haven't calibrated the camera yet, do it
    if(calibComplete == False):
        calibImage = copy.deepcopy(image)
        calibImage = calibImage[120:360, 210:430]
        calibImage = cv2.cvtColor(calibImage, cv2.COLOR_BGR2GRAY)
        calibComplete = True

    imageWithGuide = image
    #imageWithGuide = cv2.cvtColor(imageWithGuide, cv2.COLOR_BGR2GRAY)
    imageModified = copy.deepcopy(image)
    cv2.rectangle(imageWithGuide,(210, 120),(430, 360), (0,255,0),thickness=2)

    cv2.imshow("Live Feed", imageWithGuide)
    cv2.imshow("Calibrated image", calibImage)

    if(success):
        imageModified = cv2.cvtColor(imageModified, cv2.COLOR_BGR2GRAY)
        imageModified = imageModified[120:360, 210:430]
        imageModified = cv2.resize(imageModified,(150,150))
        #cv2.imshow("Image to model", imageModified)
        imageModified = np.reshape(imageModified,(1,150,150,1))

            #print(np.shape(imageModified))

        predictions = model_loaded.predict(imageModified) #include file to check out
        result = predictions[0]

        #Positive mask ID skews to 0
        # Negative on mask skews to 1
        if(result <= 0.5):
            sampleVals = sampleVals + 1
            sampleCount = sampleCount + 1
            print("Mask\n")
        else:
            sampleCount = sampleCount + 1
            print("No mask\n")

        print("Result: " , str(result))
    else:
        print("Camera did not open")
        break

    #If our sample has taken five, stop the camera
    if(sampleCount >= 5):
        break

webcam.release()
cv2.destroyAllWindows() 

#SYMPTOM CODE HERE
#
#
#
#
#
