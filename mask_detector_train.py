import cv2
import tensorflow as tf
import imutils
import numpy

BATCH_SIZE = 32
IMG_HEIGHT = 150
IMG_WIDTH = 150
dataDirectory = "/home/kevin/Desktop/AI Project/covidAI/prajna_dataset"

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataDirectory,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size = BATCH_SIZE
)

#program execution
print("\n\n\n\n\n\n")
class_names = train_dataset.class_names
print(class_names)