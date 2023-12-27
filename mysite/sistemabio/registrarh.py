#registar huellas en unbuffer
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PyFingerprint
Copyright (C) 2015 Bastian Raschke <bastian.raschke@posteo.de>
All rights reserved.

"""

import time
from pyfingerprint.pyfingerprint import PyFingerprint
from pyfingerprint.pyfingerprint import FINGERPRINT_CHARBUFFER1
from pyfingerprint.pyfingerprint import FINGERPRINT_CHARBUFFER2


## Enrolls new finger
##

## Tries to initialize the sensor
try:
    f = PyFingerprint('/dev/ttyS0', 57600, 0xFFFFFFFF, 0x00000000)

    if ( f.verifyPassword() == False ):
        raise ValueError('La contraseña del sensor de huellas dactilares proporcionada es incorrecta!')

except Exception as e:
    print('El sensor de huella no puede ser inicializado!')
    print('Exception message: ' + str(e))
    exit(1)

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
        print('La plantilla ya existe en la posición #' + str(positionNumber))
        exit(0)

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
    print('Nueva posición de plantilla #' + str(positionNumber))

except Exception as e:
    print('Operación errorea!')
    print('Exception message: ' + str(e))
    exit(1)