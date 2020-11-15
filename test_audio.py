import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_io as tfio

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

import pyaudio
import wave

def speech_recog():

    commands = ['right', 'go', 'no', 'left', 'stop', 'up', 'down','yes']
    commandsChanged = ['unknown', 'unknown' ,'no', 'unknown', 'unknown', 'unknown', 'unknown', 'yes']
    data_dir = pathlib.Path('data/mini_speech_commands')

    def decode_audio(audio_binary):
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)

        # Note: You'll use indexing here instead of tuple unpacking to enable this 
        # to work in a TensorFlow graph.
        return parts[-2] 

    def get_waveform_and_label(file_path):
        label = get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = decode_audio(audio_binary)
        return waveform, label

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    #files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    #waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    def get_spectrogram(waveform):
        # Padding for files with less than 16000 samples
        zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

        # Concatenate audio with padding so that all audio clips will be of the 
        # same length
        waveform = tf.cast(waveform, tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)
        
        spectrogram = tf.abs(spectrogram)

        return spectrogram

    def get_spectrogram_and_label_id(audio, label):
        spectrogram = get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        label_id = tf.argmax(label == commands)
        return spectrogram, label_id

    def preprocess_dataset(files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        output_ds = output_ds.map(
            get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
        return output_ds

    #Load Model
    model_loaded = tf.keras.models.load_model('speechrecog.model')

    #Get File
    #sample_file = data_dir/'yes/0ab3b47d_nohash_0.wav'
    #sample_file = pathlib.Path('ownData/yes/Audio1_yes.wav')
    sample_file = pathlib.Path('ownData/unknown/output.wav')

    # Preprocess
    sample_ds = preprocess_dataset([str(sample_file)])

    #Predict
    for spectrogram, label in sample_ds.batch(1):
        prediction = model_loaded(spectrogram)

    #print('Prediction of the model:',prediction)
    #print(' ')

    #Get array from Tensor
    array = prediction.numpy()
    #print('Array', array)
    #print(' ')

    #Get which word was said 
    higherNum = 0
    for i in array[0]:
        if i > higherNum:
            higherNum = i 
    
    if (higherNum == 0):
        return "unknown"

    index = np.where(array[0] == higherNum)
    index = index[0][0]
    #print('Index:',index)
    #print(' ')

    word = commandsChanged[index]
    return word
    #print('The word is:', word)
    #print(' ')

    #plot the prediction of the word
    #plt.bar(commandsChanged, tf.nn.softmax(prediction[0]))
    #plt.title(f'Predictions for "{word}"')
    #plt.show()

def record_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "ownData/unknown/output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()