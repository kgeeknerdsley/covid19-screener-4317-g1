import cv2
import tensorflow as tf
from tensorflow.keras import layers
import imutils
import numpy
import matplotlib.pyplot as plt

BATCH_SIZE = 32
IMG_HEIGHT = 150
IMG_WIDTH = 150
NUM_CLASSES = 2 #number of classifications, just mask / no mask
dataDirectory = "/home/kevin/Desktop/AI Project/covidAI/prajna_dataset"

EPOCHS = 10
LEARNING_RATE = 0.001

print("Loading Dataset")

#set up training dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataDirectory,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size = BATCH_SIZE
)

#set up validation dataset
validate_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  dataDirectory,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

#cache and optimize data for training
#caching data stores training data in memory, so we don't have to keep loading and unloading to disk every epoch
#prefetching allows the execution pipeline to load data for next step, right before training
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
validate_dataset = validate_dataset.cache().prefetch(buffer_size = AUTOTUNE)

print("Datasets loaded!")

#build the model!
#NOTE: CURRENTLY, THIS MODEL IS POOP. JUST FOR MAKING SURE TRAIN SCRIPT WORKS
model = tf.keras.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape = (IMG_HEIGHT,IMG_WIDTH,3)), #rescale image from 3 layer RGB, to 1 layer grayscale
    layers.Conv2D(16,3, padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16,3, padding='same',activation='relu'),
    layers.Flatten(),
    layers.Dense(16,activation='relu'),
    layers.Dense(NUM_CLASSES)
])


#compile the model
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.binary_crossentropy,
    metrics = ['accuracy']
)

#check out its structure
model.summary()

check = input("\nReady to train? Press Y or y when ready\n")

#train the dude!
fitter = model.fit(
    train_dataset,
    validation_data= validate_dataset,
    epochs = EPOCHS
)

print("\nSaving model...\n")
model.save('maskface.model')
print("Model saved!")

#Get data about training success
acc = fitter.history['accuracy']
val_acc = fitter.history['val_accuracy']

loss=fitter.history['loss']
val_loss=fitter.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('loss_accuracy.png')