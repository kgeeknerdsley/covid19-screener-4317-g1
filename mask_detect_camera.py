import cv2
import tensorflow as tf
import numpy as np
import time
import copy

secBetween = 3
startingPoint = 100

model_loaded = tf.keras.models.load_model('maskface.model')

webcam = cv2.VideoCapture(0)

#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

while(True):
    success, image = webcam.read()

    imageWithGuide = image
    imageModified = copy.deepcopy(image)
    cv2.rectangle(imageWithGuide,(startingPoint, startingPoint),(startingPoint+250, startingPoint+250), (0,255,0),thickness=2)

    cv2.imshow("Live Feed", imageWithGuide)

    if(success):
        if(cv2.waitKey(1) == 32):
            imageModified = cv2.cvtColor(imageModified, cv2.COLOR_BGR2GRAY)
            imageModified = cv2.resize(imageModified,(150,150))
            imageModified = np.reshape(imageModified,(1,150,150,1))

            #cv2.imshow("Image to model", imageModified)
            print(np.shape(imageModified))

            predictions = model_loaded.predict(imageModified) #include file to check out
            result = np.argmax(predictions)

            print("Result: " , str(result))
        elif(cv2.waitKey(1) == 27):
            print("Exiting!")
            break

    else:
        print("Camera did not open")
        break

    

webcam.release()
cv2.destroyAllWindows() 