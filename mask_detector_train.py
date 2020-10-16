import cv2
import tensorflow as tf
from tensorflow.keras import layers
import imutils
import numpy
import matplotlib.pyplot as plt

BATCH_SIZE = 64
IMG_HEIGHT = 150
IMG_WIDTH = 150
NUM_CLASSES = 2 #number of classifications, just mask / no mask
dataDirectory = "/home/kevin/Desktop/AI Project/covidAI/prajna_dataset"

EPOCHS = 55
LEARNING_RATE = 1e-6

print("Loading Dataset")

#set up training dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataDirectory,
    validation_split=0.2,
    subset="training",
    seed=123,
    shuffle=True,
    color_mode='grayscale',
    label_mode='binary',
    image_size=(IMG_HEIGHT,IMG_WIDTH),
    batch_size = BATCH_SIZE
)

#set up validation dataset
validate_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  dataDirectory,
  validation_split=0.2,
  subset="validation",
  seed=123,
  shuffle=True,
  color_mode='grayscale',
  label_mode = 'binary',
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

#cache and optimize data for training
#caching data stores training data in memory, so we don't have to keep loading and unloading to disk every epoch
#prefetching allows the execution pipeline to load data for next step, right before training
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size = AUTOTUNE)
validate_dataset = validate_dataset.cache().prefetch(buffer_size = AUTOTUNE)

print("Datasets loaded!")

#data augmentation layer
#randomly flip images, rotate a bit, and zoom a bit
data_augment_layer = tf.keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip('horizontal',
        input_shape=(IMG_HEIGHT,IMG_WIDTH,1)),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1)
    ]
)

#build the model!
#NOTE: CURRENTLY, THIS MODEL IS POOP. JUST FOR MAKING SURE TRAIN SCRIPT WORKS
model = tf.keras.Sequential([
    data_augment_layer,
    layers.experimental.preprocessing.Rescaling(1./255, input_shape = (IMG_HEIGHT,IMG_WIDTH,1)), #rescale image from 3 layer RGB, to 1 layer grayscale

    layers.Conv2D(64, (3,3), input_shape = (IMG_HEIGHT,IMG_WIDTH,1), padding='same',activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), padding='same',activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.2),

    # layers.Conv2D(64, (3,3), padding='same',activation='relu'),
    # layers.BatchNormalization(),
    # layers.Conv2D(64, (3,3), padding='same',activation='linear'),
    # layers.BatchNormalization(),
    # layers.MaxPooling2D(pool_size=(2,2)),
    # layers.Dropout(0.2),

    # layers.Conv2D(64, (3,3), padding='same',activation='relu'),
    # layers.BatchNormalization(),
    # layers.Conv2D(64, (3,3), padding='same',activation='linear'),
    # layers.BatchNormalization(),
    # layers.MaxPooling2D(pool_size=(2,2)),
    # layers.Dropout(0.2),

    # layers.Conv2D(16, (3,3), padding='same',activation='relu'),
    # layers.BatchNormalization(),
    # layers.Conv2D(64, (3,3), padding='same',activation='relu'),
    # layers.BatchNormalization(),
    # layers.MaxPooling2D(pool_size=(2,2)),
    # layers.Dropout(0.2),

    # layers.Conv2D(64, (3,3), padding='same',activation='relu'),
    # layers.BatchNormalization(),
    # layers.Conv2D(64, (3,3), padding='same',activation='linear'),
    # layers.BatchNormalization(),
    # layers.MaxPooling2D(pool_size=(2,2)),
    # layers.Dropout(0.2),

    layers.Conv2D(64, (3,3), padding='same',activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), padding='same',activation='linear'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.2),

    layers.Conv2D(64, (3,3), padding='same',activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3,3), padding='same',activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Dropout(0.2),
    
    layers.Flatten(),
    layers.Dense(4096,activation='relu'),
    layers.Dense(4096,activation='relu'),
    layers.Dense(512,activation='linear'),
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(32,activation='relu'),
    layers.Dense(16,activation='linear'),
    layers.Dense(4,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

#compile the model
model.compile(
    optimizer = tf.keras.optimizers.Adam(lr = LEARNING_RATE),
    loss = tf.keras.losses.binary_crossentropy,
    metrics = ['accuracy']
)

#check out its structure
model.summary()

lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.9,patience=3)
#stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',min_delta=0, patience=5,mode='auto')
checkpoint = tf.keras.callbacks.ModelCheckpoint('/home/kevin/Desktop/AI Project/covidAI/checkpoint.hd5', monitor='val_accuracy', verbose=1, save_best_only=True)

check = input("\nReady to train? Press Y or y when ready\n")

#train the dude!
fitter = model.fit(
    train_dataset,
    validation_data= validate_dataset,
    epochs = EPOCHS,
    shuffle=True,
    callbacks=[lr_reducer,checkpoint]
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