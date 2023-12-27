

def limpiar_lector(f):
    ## Tries to initialize the sensor
    try:
        f = PyFingerprint('/dev/ttyS0', 57600, 0xFFFFFFFF, 0x00000000)
        if ( f.verifyPassword() == False ):
            raise ValueError('La contrase�a del sensor de huellas dactilares proporcionada es incorrecta!')
    except Exception as e:
        print('El sensor de huella no puede ser inicializado!')
        print('Exception message: ' + str(e))
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
  