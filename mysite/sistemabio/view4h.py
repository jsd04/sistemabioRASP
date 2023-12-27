
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from django.shortcuts import get_object_or_404, render, redirect, get_object_or_404
from django.urls import reverse
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
from .forms import SesionFormHuella,MiFormularioSimple, SesionForm3, SesionFormVoz
import time
from pyfingerprint.pyfingerprint import PyFingerprint
from pyfingerprint.pyfingerprint import FINGERPRINT_CHARBUFFER1
from pyfingerprint.pyfingerprint import FINGERPRINT_CHARBUFFER2
import tempfile
import hashlib

# Datos biométricos
   
def huella(request,usuario_id):
     if request.method == "GET":
         inquilino = get_object_or_404(Usuario,pk=usuario_id)
         form= SesionFormHuella(instance=inquilino)
         return render(request, 'sistemabio/huella.html', 
                       {  'inquilino':inquilino,
                          "form": form
                        })
     else:
          inquilino = get_object_or_404(Usuario,pk=usuario_id)
          try:
                form = SesionFormHuella(request.POST)
                new_huella = form.save(commit=False)
                print("formulario", form.is_valid())
                inquilino = get_object_or_404(Usuario,pk=usuario_id)
                print('id usuario: ', usuario_id)
                print('usuario: ',inquilino.nombre, inquilino.ap_paterno, inquilino.ap_materno )
                personName =  str(usuario_id) + inquilino.nombre + inquilino.ap_paterno + inquilino.ap_materno 
                print("Nombre de personName es: ", personName)
                #    dataPath = 'C:/Users/yobis/Desktop/sistemabio/mysite/sistemabio/static/inquilinos' + '/' + personName
                dataPath = '/home/pi/Downloads/sistemabio/mysite/sistemabio/static/inquilinos' + '/' + personName
                personPath = dataPath + '/' + 'HUELLA' + personName
                print("Nombre de carpeta es: ", personPath)
                if not os.path.exists(personPath):
                        try:
                            os.mkdir(personPath, mode=0o755)
                            print('Carpeta creada: ',personPath)
                        except OSError as e:
                            if e.errno!=errno.EEXIST:
                                raise
                else :
                        print('el directorio ya existe')
                print('form dato: ', form['dato'].value())
                dato = form['dato'].value()
                ## Enrolls new finger
                ##
                f = inicializador_lector()
                # limpiar_lector(f)
                ## Gets some sensor information
                print('Plantillas utilizadas actualmente: ' + str(f.getTemplateCount()) +'/'+ str(f.getStorageCapacity()))
                ## Tries to enroll new finger
                try:
                    print('Esperando por una huella...')
                    ## Wait that finger is read
                    while ( f.readImage() == False ):
                        pass
                    ## Converts read image to characteristics and stores it in charbuffer 1
                    f.convertImage(FINGERPRINT_CHARBUFFER1)
                    print('Checando que la huella no haya sido registrada antes')
                    ## Checks if finger is already enrolled
                    result = f.searchTemplate()
                    positionNumber = result[0]
                    if ( positionNumber < 0 ):
                        print('La huella no existe en los registros' )
                        time.sleep(2)
                        print('Ingresa la misma huella para registrarla')
                        ## Wait that finger is read again
                        while ( f.readImage() == False ):
                            pass
                        ## Converts read image to characteristics and stores it in charbuffer 2
                        f.convertImage(FINGERPRINT_CHARBUFFER2)
                        ## Compares the charbuffers
                        if ( f.compareCharacteristics() == 0 ):
                            raise Exception('Las huellas no coinciden')
                        #ya que si coinciden vamos a descargar la huella
                        imagendescargar_huella(personPath, inquilino, f)
                        ## Creates a template
                        f.createTemplate()
                        ## Saves template at new position number
                        positionNumber = f.storeTemplate()
                        # positionNumber = f.storeTemplate()  # Obtienes la posici�n de la huella en el sensor
                        captureList = os.listdir(personPath)
                        print('lista de imagenes', captureList)
                        for filename in captureList:
                                imagepath = personPath+"/"+filename
                                print(imagepath)
                                image_file = open(personPath +'/'+filename, 'rb')
                                image = image_file.read()
                        new_huella.dato = image
                        # print(new_huella.dato)
                        new_huella.completado = True
                        print(new_huella.completado)
                        new_huella.save()
                        print('Si se guardo el formulario')
                        print('Huella registrada exitosamente!')
                        print('Nueva posici�n de plantilla #' + str(positionNumber))
                        messages.success(request, "El registro de huella ha sido un éxito.")
                        return redirect('/sistemabio/inquilinos/')
                    else :
                        print('La plantilla ya existe en la posici�n #' + str(positionNumber))
                        messages.error(request, "Error no se creo el registro de huella. La plantilla ya existe en la posici�n # " + str(positionNumber))
                        return render(request, 'sistemabio/huella.html', 
                                        { 'inquilino': inquilino,"form":  form })
                        
                except Exception as e:
                    print('Operaci�n errorea!')
                    print('Exception message: ' + str(e))
                    messages.error(request, "Error no se creo el registro de huella.")
                    return render(request, 'sistemabio/huella.html', 
                                    { 'inquilino': inquilino,"form":  form , 
                                    "error": "Error creando el registro de huella."})

          except ValueError:
               messages.error(request, "Error no se creo el registro de huella.")
               return render(request, 'sistemabio/huella.html', 
                              { 'inquilino': inquilino,"form":  form , 
                              "error": "Error creando el registro de huella."})
     
     # title='huella'
     # return render (request,'sistemabio/vozjj.html',{
     #      'mytitle':title
     # })

def inicializador_lector():
    ## Tries to initialize the sensor
    try:
        f = PyFingerprint('/dev/ttyS0', 57600, 0xFFFFFFFF, 0x00000000)
        if ( f.verifyPassword() == False ):
            raise ValueError('La contrase�a del sensor de huellas dactilares proporcionada es incorrecta!')
    except Exception as e:
        print('El sensor de huella no puede ser inicializado!')
        print('Exception message: ' + str(e))
        #  exit(1)
    return f

def imagendescargar_huella(personPath, inquilino, f):
     ## Reads image and download it ##
     ## Tries to read image and download it
     try:
        while ( f.readImage() == False ):
            pass
        print('Descargando imagen ( espera un momento )...')
        # imageDestination =  tempfile.gettempdir() + '/fingerprint.bmp'
        huella_name = 'huella_user_'+ str(inquilino.id_usuario) +'.bmp' # Guardar los datos d en un archivo
        print('huella_name: ',huella_name)
        imageDestination =  personPath + '/'+ huella_name
        f.downloadImage(imageDestination)
        print('La imagen fue guardada en "' + imageDestination + '".')
     except Exception as e:
        print('la descarga de huella fallo !')
        print('Exception message: ' + str(e))

def limpiar_lector(f):
    # Gets some sensor information
    print('Currently used templates: ' + str(f.getTemplateCount()) + '/' + str(f.getStorageCapacity()))

    # Tries to clear the entire fingerprint database
    try:
        confirm = input('Do you really want to clear the entire fingerprint database? (yes/no): ').lower()
        if confirm == 'yes':
            f.clearDatabase()
            print('Fingerprint database cleared!')
        else:
            print('Operation aborted.')

    except Exception as e:
        print('Operation failed!')
        print('Exception message: ' + str(e))
        
def registrar_huella():
    
     ## Enrolls new finger
     ##

     ## Tries to initialize the sensor
     try:
         f = PyFingerprint('/dev/ttyS0', 57600, 0xFFFFFFFF, 0x00000000)

         if ( f.verifyPassword() == False ):
             raise ValueError('La contrase�a del sensor de huellas dactilares proporcionada es incorrecta!')

     except Exception as e:
         print('El sensor de huella no puede ser inicializado!')
         print('Exception message: ' + str(e))
        #  exit(1)

     ## Gets some sensor information
     print('Plantillas utilizadas actualmente: ' + str(f.getTemplateCount()) +'/'+ str(f.getStorageCapacity()))

     ## Tries to enroll new finger
     try:
         print('Esperando por una huella...')

         ## Wait that finger is read
         while ( f.readImage() == False ):
             pass

         ## Converts read image to characteristics and stores it in charbuffer 1
         f.convertImage(FINGERPRINT_CHARBUFFER1)

         ## Checks if finger is already enrolled
         result = f.searchTemplate()
         positionNumber = result[0]

         if ( positionNumber >= 0 ):
             print('La plantilla ya existe en la posici�n #' + str(positionNumber))
            #  exit(0)

         print('Remove finger...')
         time.sleep(2)

         print('Esperando por una huella otra vez...')

         ## Wait that finger is read again
         while ( f.readImage() == False ):
             pass

         ## Converts read image to characteristics and stores it in charbuffer 2
         f.convertImage(FINGERPRINT_CHARBUFFER2)

         ## Compares the charbuffers
         if ( f.compareCharacteristics() == 0 ):
             raise Exception('Las huellas no coinciden')

         ## Creates a template
         f.createTemplate()

         ## Saves template at new position number
         positionNumber = f.storeTemplate()
         print('Huella registrada exitosamente!')
         print('Nueva posici�n de plantilla #' + str(positionNumber))

     except Exception as e:
         print('Operaci�n errorea!')
         print('Exception message: ' + str(e))
        #  exit(1)

        
def buscar_huella():
     ## Search for a finger ##
     ## Tries to initialize the sensor
     try:
        f = PyFingerprint('/dev/ttyS0', 57600, 0xFFFFFFFF, 0x00000000)

        if ( f.verifyPassword() == False ):
            raise ValueError('The given fingerprint sensor password is wrong!')

     except Exception as e:
        print('The fingerprint sensor could not be initialized!')
        print('Exception message: ' + str(e))
        exit(1)

     ## Gets some sensor information
     print('Currently used templates: ' + str(f.getTemplateCount()) +'/'+ str(f.getStorageCapacity()))

     ## Tries to search the finger and calculate hash
     try:
        print('Waiting for finger...')

        ## Wait that finger is read
        while ( f.readImage() == False ):
            pass

        ## Converts read image to characteristics and stores it in charbuffer 1
        f.convertImage(FINGERPRINT_CHARBUFFER1)

        ## Searchs template
        result = f.searchTemplate()

        positionNumber = result[0]
        accuracyScore = result[1]

        if ( positionNumber == -1 ):
            print('No match found!')
            exit(0)
        else:
            print('Found template at position #' + str(positionNumber))
            print('The accuracy score is: ' + str(accuracyScore))
        ## OPTIONAL stuff

        ##

        ## Loads the found template to charbuffer 1
        f.loadTemplate(positionNumber, FINGERPRINT_CHARBUFFER1)

        ## Downloads the characteristics of template loaded in charbuffer 1
        characterics = str(f.downloadCharacteristics(FINGERPRINT_CHARBUFFER1)).encode('utf-8')

        ## Hashes characteristics of template
        print('SHA-2 hash of template: ' + hashlib.sha256(characterics).hexdigest())

     except Exception as e:
        print('Operation failed!')
        print('Exception message: ' + str(e))
        exit(1)

def accediste(request):
    title='accediste'
    return render (request,'sistemabio/accediste.html',{
         'mytitle':title
    })

def huella_usuario(request):
    if request.method == "GET":
        form = MiFormularioSimple()
        return render(request, 'sistemabio/huella_usuario.html', {
            "form": form
        })
    else:
        form = MiFormularioSimple(request.POST)
        # new_facial_usuario = form.save(commit=False)
        print('form : ', form['dato_simple'].value())
        
        personPath = '/home/pi/Downloads/sistemabio/mysite/sistemabio/static/newusuarioh'
        personPath_2 = '/home/pi/Downloads/sistemabio/mysite/sistemabio/static/inquilinos'
        dato_simple = form['dato_simple'].value()
        f = inicializador_lector()
        ## Gets some sensor information
        print('Plantillas utilizadas actualmente: ' + str(f.getTemplateCount()) +'/'+ str(f.getStorageCapacity()))
        ## Search for a finger ##
        try:
            print('Esperando por una huella...')
            ## Wait that finger is read
            while ( f.readImage() == False ):
                pass
            print('Descargando imagen ( espera un momento )...')
            new_huella_name = 'huella_new_user.bmp' # Guardar los datos d en un archivo
            print('new_huella_name: ',new_huella_name)
            new_imageDestination =  personPath + '/'+ new_huella_name
            f.downloadImage(new_imageDestination)
            print('La imagen fue guardada en "' + new_imageDestination + '".')
            ## Converts read image to characteristics and stores it in charbuffer 1
            f.convertImage(FINGERPRINT_CHARBUFFER1)
            print('buscando la huella ingresada ...')
            ## Searchs template
            result = f.searchTemplate()
            positionNumber = result[0]
            accuracyScore = result[1]
            if ( positionNumber == -1 ):
                print('No se encontraron coincidencias!')
                messages.error(request, "Error no se reconocio la huella en los registros.")
                return render(request, 'sistemabio/huella_usuario.html', 
                        { "form":  form })
            else:
                print('Huella encontrada en la posici�n #' + str(positionNumber))
                print('La puntuaci�n de precisi�n es: ' + str(accuracyScore))
                ## OPTIONAL stuff ##
                ## Loads the found template to charbuffer 1
                f.loadTemplate(positionNumber, FINGERPRINT_CHARBUFFER1)
                ## Downloads the characteristics of template loaded in charbuffer 1
                characterics = str(f.downloadCharacteristics(FINGERPRINT_CHARBUFFER1)).encode('utf-8')
                ## Hashes characteristics of template
                print('SHA-2 hash of template: ' + hashlib.sha256(characterics).hexdigest())
                messages.success(request, "El reconocimiento de huella dactilar ha sido un éxito.")
                return redirect('/sistemabio/accediste/')
        except ValueError:
          messages.error(request, "Error no se tomo la huella correctamente.")
          return render(request, 'sistemabio/huella.html',{ "form":  form })
        
    # title='huella_usuario'
    # return render (request,'sistemabio/huella_usuario.html',{
    #      'mytitle':title
    # })




