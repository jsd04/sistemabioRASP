
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
# from .forms import InquilinoForm, SesionForm, SesionFormUsuario,MiFormularioSimple, SesionForm3, SesionFormVoz
from .forms import MiFormularioSimple, SesionForm3, SesionFormVoz
import subprocess
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
    # plt.figure(1)
    # pd.Series(y).plot(figsize=(10, 5),
    #                 lw=1,
    #                 title='Raw Audio ',
    #                 color=color_pal[0])
    print('si leee')
    # Trimming leading/lagging silence
    # plt.figure(2)  
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    # pd.Series(y_trimmed).plot(figsize=(10, 5),
    #                 lw=1,
    #                 title='Raw Audio Trimmed ',
    #                 color=color_pal[1])
    # plt.show()
    audio_trimmed_path = personPath + "/" + filename  # Guardar la señal en un archivo WAV)
    wavfile.write(audio_trimmed_path, s_r, y_trimmed)
    print(f"Archivo de audio recortado y guardado en: {audio_trimmed_path}")
    return audio_trimmed_path

def coeficientes(audio_trimmed_path):
    print('noss')

def archivo_csv (audio_trimmed_path,inquilino):
    archivo_csv = "PruebaProyect.csv"
    # csv_path =   'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/inquilinos/' + archivo_csv
    csv_path =   '/home/pi/Downloads/sistemabio/mysite/sistemabio/static/inquilinos/' + archivo_csv
    audio = audio_trimmed_path
    columnas = ["ID", "NOMBRE", "VOZ"]
    print("inquilino: ",inquilino)
    nameperson = inquilino.nombre + inquilino.ap_paterno + inquilino.ap_materno
    datos = [str(inquilino.id_usuario), nameperson, audio]
    if os.path.exists(csv_path):
        with open(csv_path, "a", newline='') as archivo:# Abrir el archivo en modo de escritura (append)
            data = csv.writer(archivo, delimiter=',')
            data.writerow(datos) # Escribir los datos en el archivo existente
            print('Se añadio información correctamente')
        with open(csv_path, "r", newline='') as archivo:
            reader = csv.reader(archivo, delimiter=',') # Leer el contenido del archivo después de escribir en él
            lectura = list(reader)
            print(lectura)
    else:
        with open(csv_path, "w", newline='') as archivo:
            data = csv.writer(archivo, delimiter=',') # Crear un nuevo archivo y escribir las columnas y datos
            data.writerow(columnas)
            data.writerow(datos)
            print('Se creo el archivo y añadio información correctamente')
        with open(csv_path, "r", newline='') as archivo:
            reader = csv.reader(archivo, delimiter=',')
            lectura = list(reader)
            print('Archivo creado y Datos: ', lectura)

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
            personPath_copia = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/voz/' + 'fold'+ str(usuario_id)
            if not os.path.exists(personPath_copia):
                try:
                    os.makedirs(personPath_copia, mode=0o755)
                    print('Carpeta cop�a creada:', personPath_copia)
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
                #guardar copia
                output_wav_path_copia = personPath_copia + "/" + filename
                audio.export(output_wav_path_copia, format="wav") 
                print(f"El archivo copia se ha guardado en: {output_wav_path_copia}")
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
                            # Leer el archivo limpio en formato binario
                            with open(audio_trimmed_path, "rb") as audio_file:
                                audio_bytes = audio_file.read()
                            datos_codificados = base64.b64encode(audio_bytes) # Codificar en base64
                            new_voz.dato = datos_codificados
                            new_voz.completado = True
                            print(new_voz.completado)
                            new_voz.save()
                            print('Si se guardo el formulario')
                            archivo_csv (audio_trimmed_path,inquilino)
                            # coeficientes(audio_trimmed_path)
                            #  # Calcular los coeficientes MFCC
                            #  extract_mfcc(audio_trimmed_path)
                            #  #Extraer las mediciones del audio.
                            #  extract_features(audio_trimmed_path)
                            #  # Librería pyAudioAnalysis realizar la extracción (low-term) de varias características de una señal de audio
                            #  extraccion(audio_trimmed_path)
                            #cnn_voz(personPath)
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
    print('inicio de reconocimiento prediccion ')

    print('finalizaciom de reconocimiento prediccion VOZ Tryue ')
    print('finalizaciom de reconocimiento prediccion VOZ False')
    return title
 
def voz_usuario(request):
    if request.method == "GET":
        form = MiFormularioSimple()
        return render(request, 'sistemabio/voz_usuario.html', {
            "form": form
        })
    else:
        # form = MiFormularioSimple(request.POST)
        form = MiFormularioSimple(request.POST)
        # new_facial_usuario = form.save(commit=False)
        print('form : ', form['dato_simple'].value())
        if form.is_valid():
            # personPath = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/newusuariov'
            personPath = '/home/pi/Downloads/sistemabio/mysite/sistemabio/static/newusuariov'
            # personPath_2 = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/inquilinos'
            personPath_2 = '/home/pi/Downloads/sistemabio/mysite/sistemabio/static/inquilinos'
            dato_simple = form['dato_simple'].value()
            # Decodificar la cadena Base64
            datos_decodificados = base64.b64decode(dato_simple)
            # Guardar los datos de audio en un archivo
            voz_name = 'audio_nuevo_usuario.wav'
            print('voz_name: ',voz_name)
            # Ahora, tienes los datos de audio en su formato original en el archivo "audio_original.wav"
            with open(personPath+'/'+voz_name, 'wb') as audio_file:
                 audio_file.write(datos_decodificados)
            print('audio guardado')
            # metodo = 2
            # new_voz = 0
            # inquilino = 0
            # audio_cleaned_path = det_recognize(personPath, new_voz, metodo, inquilino)
            # audio_cleaned_path = det_recognize(personPath)
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
                 # Leer el archivo WAV
                sample_rate, audio_data = wav.read(output_wav_path)
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
                            # segun yo ya no genera csv ni extrae coeficientes  ni red, solo predicion
                            # archivo_csv (audio_trimmed_path,inquilino)
                            # coeficientes(audio_trimmed_path)
                            #  # Calcular los coeficientes MFCC
                            #  extract_mfcc(audio_trimmed_path)
                            #  #Extraer las mediciones del audio.
                            #  extract_features(audio_trimmed_path)
                            #  # Librería pyAudioAnalysis realizar la extracción (low-term) de varias características de una señal de audio
                            #  extraccion(audio_trimmed_path)
                            # cnn_voz(personPath)
                            print('si termino voz') 
                            #Aqui mismo hacer reconocimiento
                            if(reconocimiento_voz(audio_trimmed_path,personPath_2)):
                                messages.success(request, "El reconocimiento de voz ha sido un éxito.")
                                return redirect('/sistemabio/accediste/')
                            else:
                                messages.error(request, "Error no se reconocio la voz dentro de la nase de datos.")
                                return render(request, 'sistemabio/voz_usuario.html', 
                                        { "form":  form, 
                                        "error": "Error creando el registro de voz."})   
                        else:
                             print("no es la frase")
                             messages.error(request, "Error: Revisa que la frase sea correcta.")
                             return render(request, 'sistemabio/voz_usuario.html',{"form": form,"error": "Error creando el registro de voz."})
                    except sr.UnknownValueError:
                        print(f"No se pudo reconocer la voz en el archivo {filename}.")
            
           

    # title='voz_usuario'
    # return render (request,'sistemabio/voz_usuario.html',{
    #      'mytitle':title
    # })

    



