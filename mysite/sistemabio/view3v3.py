
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from django.http import HttpResponseRedirect,HttpResponse, Http404
from django.shortcuts import get_object_or_404, render, redirect, get_object_or_404
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.template import loader
import datetime
import base64
import os
import errno
import cv2
import cv2 as cv
import time
import re
import noisereduce as nr
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io.wavfile as wav
import librosa
from pydub import AudioSegment
import speech_recognition as sr
from scipy.io import wavfile
import torch
import torchaudio
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
import math
import csv
from glob import glob
import librosa
import librosa.display
from itertools import cycle
import numpy as np
import pylab as plt
from scipy.fftpack import dct
from django.contrib import messages
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adagrad
from django.contrib.auth.decorators import login_required
from .models import Usuario, Sesion
from .forms import MiFormularioSimple, SesionForm3, SesionFormVoz
import subprocess
import os
import pathlib
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report
from keras import layers
from keras import models
from IPython import display

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

 
# Datos biométricos

def audioTrimmed(personPath,filename,output_wav_path):
    # Leer el archivo WAV
    y, s_r = librosa.load(output_wav_path)
    print(f'y: {y[:10]}')
    print(f'shape y: {y.shape}')
    print(f'sr: {s_r}')
    print('si leee')
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    audio_trimmed_path = personPath + "/" + filename  # Guardar la señal en un archivo WAV)
    wavfile.write(audio_trimmed_path, s_r, y_trimmed)
    print(f"Archivo de audio recortado y guardado en: {audio_trimmed_path}")
    return audio_trimmed_path       

def copia_audio(audio_trimmed_path,personPath_copia, filename):
    print("COPIAS")
    # Leer el archivo WAV
    y, s_r = librosa.load(audio_trimmed_path)
    print(f'y: {y[:10]}')
    print(f'shape y: {y.shape}')
    print(f'sr: {s_r}')
    print('si leee')
    # audio = AudioSegment.from_file(audio_trimmed_path) # Cargar el archivo de audio en formato x
                # output_wav_path = personPath + "/" + filename # Ruta de salida para el archivo WAV
                # audio.export(output_wav_path, format="wav") # Exportar el archivo a formato WAV
    num_saves=50
    # audio_trimmed_path = personPath + "/" + filename  # Guardar la señal en un archivo WAV)
    # wavfile.write(audio_trimmed_path, s_r, y_trimmed)
    # print(f"Archivo de audio recortado y guardado en: {audio_trimmed_path}")
    # Separate filename and extension
    name, ext = os.path.splitext(filename)
    print("namedel archivo: ",name)
    # # Convert the original audio to 16-bit
    # save_as_16_bit_wav(y, s_r, audio_trimmed_path)
    for i in range(num_saves):
        # Append an index to the filename to make each save unique
        save_filename = f"{name}_copy{i+1}.wav"

        # audio_trimmed_copia = personPath_copia + "/" + save_filename
   
        # wavfile.write(audio_trimmed_copia, s_r, y)
        audio_trimmed_copia = personPath_copia + "/" + save_filename  # Ruta de salida para el archivo WAV
        # audio.export(audio_trimmed_copia, format="wav") # Exportar el archivo a formato WAV
        audio_16_bit = (y * 32767).astype(np.int16)
        wavfile.write(audio_trimmed_copia, s_r, audio_16_bit)
        # save_as_16_bit_wav(y, s_r, audio_trimmed_copia)
        print(f"El archivo copia recortado se ha guardado en: {audio_trimmed_copia}")

    return audio_trimmed_copia
def red_voz(request):
    title='red voz'
    if request.method == 'POST':
          print('entrenamiento: ')
          try:
               train_audio_model()
               messages.success(request, "¡El entrenamiento se hizo correctamente!")
               return redirect('/sistemabio/inquilinos/')
          except ValueError:
               messages.error(request, "Error al entrenar la red de voz.")
               return render(request,"sistemabio/red_voz.html")
    else:
          return render(request,"sistemabio/red_voz.html")


def train_audio_model():
    # Set the seed value for experiment reproducibility.
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    training_set, validation_set = tf.keras.utils.audio_dataset_from_directory(
        # directory='C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/Voz',
        directory='/home/pi/Downloads/sistemabio/mysite/sistemabio/Voz',
        batch_size=44,
        validation_split=0.2,
        output_sequence_length=16000,
        seed=0,
        subset='both'
    )

    label_names = np.array(training_set.class_names)
    print("label names:", label_names)

    def squeeze(audio, labels):
        audio = tf.squeeze(audio, axis=-1)
        return audio, labels

    training_set = training_set.map(squeeze, tf.data.AUTOTUNE)
    validation_set = validation_set.map(squeeze, tf.data.AUTOTUNE)

    audio, label = next(iter(training_set))
    display.display(display.Audio(audio[0], rate=16000))

    # ... (rest of the code block)
        # Plot the waveform 
    def plot_wave(waveform, label): 
        plt.figure(figsize=(10, 3)) 
        plt.title(label) 
        plt.plot(waveform) 
        plt.xlim([0, 16000]) 
        plt.ylim([-1, 1]) 
        plt.xlabel('Time') 
        plt.ylabel('Amplitude') 
        plt.grid(True) 
    
    # Convert waveform to spectrogram 
    def get_spectrogram(waveform): 
        spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128) 
        spectrogram = tf.abs(spectrogram) 
        return spectrogram[..., tf.newaxis] 
    
    # Plot the spectrogram 
    def plot_spectrogram(spectrogram, label): 
        spectrogram = np.squeeze(spectrogram, axis=-1) 
        log_spec = np.log(spectrogram.T + np.finfo(float).eps) 
        plt.figure(figsize=(10, 3)) 
        plt.title(label) 
        plt.imshow(log_spec, aspect='auto', origin='lower') 
        plt.colorbar(format='%+2.0f dB') 
        plt.xlabel('Time') 
        plt.ylabel('Frequency') 

    # Plotting the waveform and the spectrogram of a random sample 
    audio, label = next(iter(training_set)) 
    
    # Plot the wave with its label name 
    plot_wave(audio[0], label_names[label[0]]) 
    
    # Plot the spectrogram with its label name 
    plot_spectrogram(get_spectrogram(audio[0]), label_names[label[0]])
    
    # Creating spectrogram dataset from waveform or audio data 
    def get_spectrogram_dataset(dataset): 
        dataset = dataset.map( 
            lambda x, y: (get_spectrogram(x), y), 
            num_parallel_calls=tf.data.AUTOTUNE) 
        return dataset 

    # Applying the function on the audio dataset 
    train_set = get_spectrogram_dataset(training_set) 
    validation_set = get_spectrogram_dataset(validation_set) 

    # Dividing validation set into two equal val and test set 
    val_set = validation_set.take(validation_set.cardinality() // 2) 
    test_set = validation_set.skip(validation_set.cardinality() // 2)

    train_set_shape = train_set.element_spec[0].shape 
    val_set_shape = val_set.element_spec[0].shape 
    test_set_shape = test_set.element_spec[0].shape 

    print("Train set shape:", train_set_shape) 
    print("Validation set shape:", val_set_shape) 
    print("Testing set shape:", test_set_shape)

    # Defining the model 
    def get_model(input_shape, num_labels): 
        model = tf.keras.Sequential([ 
            tf.keras.layers.Input(shape=input_shape), 
            # Resizing the input to a square image of size 64 x 64 and normalizing it 
            tf.keras.layers.Resizing(64, 64), 
            tf.keras.layers.Normalization(), 

            # Convolution layers followed by MaxPooling layer 
            tf.keras.layers.Conv2D(64, 3, activation='relu'), 
            tf.keras.layers.Conv2D(128, 3, activation='relu'), 
            tf.keras.layers.MaxPooling2D(), 
            tf.keras.layers.Dropout(0.5), 
            tf.keras.layers.Flatten(), 

            # Dense layer 
            tf.keras.layers.Dense(256, activation='relu'), 
            tf.keras.layers.Dropout(0.5), 

            # Softmax layer to get the label prediction 
            tf.keras.layers.Dense(num_labels, activation='softmax') 
        ]) 
        # Printing model summary 
        model.summary() 
        return model 

    # Getting input shape from the sample audio and number of classes 
    input_shape = next(iter(train_set))[0][0].shape 
    print("Input shape:", input_shape) 
    num_labels = len(label_names) 

    # Creating a model 
    model = get_model(input_shape, num_labels)

    model.compile( 
        optimizer=tf.keras.optimizers.Adam(), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        metrics=['accuracy'], 
    ) 
    print("compila modelo")

    EPOCHS = 10
    history = model.fit(
        train_set,
        validation_data=val_set,
        epochs=EPOCHS,
    )

    y_pred = np.argmax(model.predict(test_set), axis=1)
    y_true = np.concatenate([y for x, y in test_set], axis=0)
    cm = tf.math.confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    report = classification_report(y_true, y_pred)
    print(report)
    model.save("pruebaVoz1.h5py")

def voz3(request, usuario_id):
    if request.method == "GET":
        inquilino = get_object_or_404(Usuario, pk=usuario_id)
        form = SesionFormVoz(instance=inquilino)
        return render(request, 'sistemabio/vozjj.html', {'inquilino': inquilino,"form": form})
    else:
        form = SesionFormVoz(request.POST)
        new_voz = form.save(commit=False)
        if form.is_valid():
            print("formulario", form.is_valid())
            dato = form['dato'].value()
            inquilino = get_object_or_404(Usuario, pk=usuario_id)
            personName = str(usuario_id) + inquilino.nombre + inquilino.ap_paterno + inquilino.ap_materno
            print("Nombre de personName es: ", personName)
            # dataPath = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/inquilinos/' + personName
            dataPath = '/home/pi/Downloads/sistemabio/mysite/sistemabio/static/inquilinos/' + personName
            personPath = dataPath + '/' + 'VOZ' + personName
            print("Nombre de carpeta es: ", personPath)
            if not os.path.exists(personPath):
                try:
                    os.makedirs(personPath, mode=0o755)
                    print('Carpeta creada:', personPath)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            else:
                print('el directorio ya existe')
            # copia de la carpeta
            # personPath_copia = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/Voz/' + 'fold'+ str(usuario_id)
            personPath_copia = '/home/pi/Downloads/sistemabio/mysite/sistemabio/Voz/' + 'fold'+ str(usuario_id)
            if not os.path.exists(personPath_copia):
                try:
                    os.makedirs(personPath_copia, mode=0o755)
                    print('Carpeta copía creada:', personPath_copia)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            else:
                print('el directorio ya existe')
            datos_decodificados = base64.b64decode(dato) # Decodificar la cadena Base64
            voz_name = 'audio_user_'+ str(usuario_id) +'.wav' # Guardar los datos de audio en un archivo
            print('voz_name: ',voz_name)
            # Los datos de audio en su formato original en el archivo "audio_original.wav"
            with open(personPath+'/'+voz_name, 'wb') as audio_file:
                 audio_file.write(datos_decodificados)
            print('audio guardado')
            # metodo = 1
            # det_recognize(personPath, new_voz, metodo, inquilino) 
            captureList = os.listdir(personPath)
            print('lista de voices', captureList)
            voz_array = []
            for filename in captureList:
                vozpath = personPath+"/"+filename
                print(vozpath)
                audio = AudioSegment.from_file(vozpath) # Cargar el archivo de audio en formato x
                output_wav_path = personPath + "/" + filename # Ruta de salida para el archivo WAV
                audio.export(output_wav_path, format="wav") # Exportar el archivo a formato WAV
                print(f"El archivo ha sido convertido a WAV y guardado en: {output_wav_path}")
                # #guardar copia
                # output_wav_path_copia = personPath_copia + "/" + filename
                # audio.export(output_wav_path_copia, format="wav") 
                # print(f"El archivo copia se ha guardado en: {output_wav_path_copia}")
                 # Leer el archivo WAV
                sample_rate, audio_data = wav.read(output_wav_path)
                print('si leee')
                # Convertir a formato mono si es estéreo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                frase_especifica = "esto es una prueba" # Frase específica que deseas comparar acceso al edificio 
                # Identificar la voz humana usando SpeechRecognition
                r = sr.Recognizer()
                with sr.AudioFile(vozpath) as source:
                    audio_text = r.record(source)
                    try:
                        audio_trimmed_path = audioTrimmed(personPath,filename,output_wav_path)
                        # recognized_text = r.recognize_google(audio_text, language='es-MX', show_all=False)
                        recognized_text = r.recognize_google(audio_text, language='es-MX')
                        print(f"Texto reconocido del archivo {filename}: {recognized_text}")
                        # Comparar con la frase específica
                        if recognized_text.lower() == frase_especifica.lower():
                            print("Sí, es la frase correcta.")
                            copia_audio(audio_trimmed_path,personPath_copia,filename)
                            # Leer el archivo limpio en formato binario
                            with open(audio_trimmed_path, "rb") as audio_file:
                                audio_bytes = audio_file.read()
                            datos_codificados = base64.b64encode(audio_bytes) # Codificar en base64
                            new_voz.dato = datos_codificados
                            new_voz.completado = True
                            print(new_voz.completado)
                            new_voz.save()
                            print('Si se guardo el formulario')
                            # train_audio_model()
                            # print("inicio y acabo red")
                            
                            print('si termino voz')
                            messages.success(request, "El registro de voz ha sido un éxito.")
                            return redirect('/sistemabio/inquilinos/')
                        else:
                             messages.error(request, "Error: Revisa que la frase sea correcta.")
                             return render(request, 'sistemabio/vozjj.html',{'inquilino': inquilino,"form": form,"error": "Error creando el registro de voz."})
                    except sr.UnknownValueError:
                        print(f"No se pudo reconocer la voz en el archivo {filename}.")
        else:
            messages.error(request, "Error: no se creó el registro de voz.")
            return render(request, 'sistemabio/vozjj.html', {'inquilino': inquilino,"form": form,"error": "Error creando el registro de voz."})
           
def reconocimiento_voz(audio_cleaned_path,personPath_2):
    title='esto retorna un true o false'
    print("el audio esta : ", audio_cleaned_path)
    print("el audio esta es: ", personPath_2)
    print('inicio de reconocimiento prediccion ')
    def get_spectrogram(waveform): 
        spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128) 
        spectrogram = tf.abs(spectrogram) 
        return spectrogram[..., tf.newaxis] 
    # Defining the squeeze function 
    def squeeze(audio, labels): 
        audio = tf.squeeze(audio, axis=-1) 
        return audio, labels 
    model = keras.models.load_model('pruebaVoz1.h5py')
    # Set the seed value for experiment reproducibility.
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    training_set, validation_set = tf.keras.utils.audio_dataset_from_directory(
        directory='/home/pi/Downloads/sistemabio/mysite/sistemabio/Voz',
        batch_size=44,
        validation_split=0.2,
        output_sequence_length=16000,
        seed=0,
        subset='both'
    )

    label_names = np.array(training_set.class_names)
    print("label names:", label_names)

    # path = 'SETS/jess0 - copia (4).wav'
    path = audio_cleaned_path
    print("el archivo a reconocer es1: ",audio_cleaned_path)
    Input = tf.io.read_file(str(audio_cleaned_path)) 
    x, sample_rate = tf.audio.decode_wav(Input, desired_channels=1, desired_samples=16000,) 
    audio, labels = squeeze(x, 'jessica') 
    print("el archivo a reconocer es2: ",audio_cleaned_path)

    waveform = audio 
    display.display(display.Audio(waveform, rate=16000)) 

    x = get_spectrogram(audio) 
    x = tf.expand_dims(x, axis=0) 

    prediction = model(x) 
    # plt.bar(label_names, tf.nn.softmax(prediction[0])) 
    # plt.title('Prediction : '+label_names[np.argmax(prediction, axis=1).item()]) 
    # plt.show()
    umbral_n = 0.8 
    pred_label = label_names[np.argmax(prediction, axis=1).item()]
    print("Usuario", pred_label)
    mP = np.round(np.max(prediction[0]), 2)
    print(mP)
    umbral_prob = 0.8
    if mP>=umbral_prob:
        print("Usuario indentificado: ",pred_label,"\n probabilidad: ",mP )
        print('finalizacion de reconocimiento prediccion VOZ True')
        return True,pred_label
    else: 
        print("Usuario NO indentificado: ",pred_label,"\n probabilidad: ",mP )
        print('finalizacion de reconocimiento prediccion VOZ False')
        return False,pred_label

def audio_16(audio_cleaned_path,personPath_2):
    print("16")
    # Leer el archivo WAV
    print("el audio esta : ", audio_cleaned_path)
    y, s_r = librosa.load(audio_cleaned_path)
    print(f'y: {y[:10]}')
    print(f'shape y: {y.shape}')
    print(f'sr: {s_r}')
    print('si leee')
   
    # name, ext = os.path.splitext(filename)
    # print("namedel archivo: ",name)
    # # Convert the original audio to 16-bit
    # Append an index to the filename to make each save unique
    # save_filename = f"{name}_copy.wav"
    save_filename = f"audio_nuevo_usuario_copy.wav"
    # audio_16_copia_path = audio_cleaned_path + "/" + save_filename  # Ruta de salida para el archivo WAV
    audio_16_copia_path = audio_cleaned_path # Ruta de salida para el archivo WAV
    # audio.export(audio_trimmed_copia, format="wav") # Exportar el archivo a formato WAV
    audio_16_bit = (y * 32767).astype(np.int16)
    wavfile.write(audio_16_copia_path, s_r, audio_16_bit)
    # save_as_16_bit_wav(y, s_r, audio_trimmed_copia)
    print(f"El archivo copia recortado se ha guardado en: {audio_16_copia_path}")

    return audio_16_copia_path


def voz_usuario(request):
    if request.method == "GET":
        form = MiFormularioSimple()
        return render(request, 'sistemabio/voz_usuario.html', {
            "form": form
        })
    else:
        form = MiFormularioSimple(request.POST)
        # print('form : ', form['dato_simple'].value())
        # red()
        # train_audio_model()
        # print("inicio y acabo red")
        if form.is_valid():
            # personPath = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/newusuariov'
            # personPath_2 = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/inquilinos'
            personPath = '/home/pi/Downloads/sistemabio/mysite/sistemabio/static/newusuariov'
            personPath_2 = '/home/pi/Downloads/sistemabio/mysite/sistemabio/static/inquilinos'
            dato_simple = form['dato_simple'].value()
            # Decodificar la cadena Base64
            datos_decodificados = base64.b64decode(dato_simple)
            voz_name = 'audio_nuevo_usuario.wav'
            print('voz_name: ',voz_name)
            # Ahora, tienes los datos de audio en su formato original en el archivo "audio_original.wav"
            with open(personPath+'/'+voz_name, 'wb') as audio_file:
                 audio_file.write(datos_decodificados)
            print('audio guardado')
            captureList = os.listdir(personPath)
            print('lista de voices', captureList)
            voz_array = []
            for filename in captureList:
                vozpath = personPath+"/"+filename
                print(vozpath)
                audio = AudioSegment.from_file(vozpath) # Cargar el archivo de audio en formato x
                output_wav_path = personPath + "/" + filename # Ruta de salida para el archivo WAV
                audio.export(output_wav_path, format="wav") # Exportar el archivo a formato WAV
                print(f"El archivo ha sido convertido a WAV y guardado en: {output_wav_path}")
                sample_rate, audio_data = wav.read(output_wav_path) # Leer el archivo WAV
                print('si leee')
                # Convertir a formato mono si es estéreo
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                frase_especifica2 = "esto es una prueba" # Frase específica que deseas comparar
                # Identificar la voz humana usando SpeechRecognition
                r = sr.Recognizer()
                with sr.AudioFile(vozpath) as source:
                    audio_text = r.record(source)
                    try:
                        audio_trimmed_path = audioTrimmed(personPath,filename,output_wav_path)
                        recognized_text = r.recognize_google(audio_text, language='es-MX', show_all=False)
                        print(f"Texto reconocido del archivo {filename}: {recognized_text}")
                        # Comparar con la frase específica
                        if recognized_text.lower() == frase_especifica2.lower():
                            print("Sí, es la frase correcta.")                           
                            #Aqui mismo hacer reconocimiento
                            audio16_path=audio_16(audio_trimmed_path,personPath_2)
                            resultado, persona =reconocimiento_voz(audio16_path,personPath_2)
                            print("resultado: ", resultado)
                            print("persona: ", persona)
                            if(resultado == True):
                                messages.success(request, "El reconocimiento de voz ha sido un éxito.")
                                return redirect('/sistemabio/accediste/')
                            else:
                                messages.error(request, "Error no se reconocio la voz dentro de la base de datos.")
                                return render(request, 'sistemabio/voz_usuario.html', 
                                        { "form":  form, 
                                        "error": "Error creando el registro de voz."}) 
                        else:
                             print("no es la frase")
                             messages.error(request, "Error: Revisa que la frase sea correcta.")
                             return render(request, 'sistemabio/voz_usuario.html',{"form": form,"error": "Error creando el registro de voz."})
                    except sr.UnknownValueError:
                        print(f"No se pudo reconocer la voz en el archivo {filename}.")
        else :
            messages.error(request, "Error: No se ha llenado el formulario.")
            return render(request, 'sistemabio/voz_usuario.html',{"form": form})    
           


    



