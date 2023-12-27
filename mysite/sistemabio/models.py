
# Create your models here.
import datetime
from django.db import models
from django.utils import timezone
from django.contrib import admin

# Create your models here.

class Usuario(models.Model):
    id_usuario = models.AutoField(primary_key=True)
    nombre = models.CharField(max_length=30)
    ap_paterno = models.CharField(max_length=15)
    ap_materno = models.CharField(max_length=15)
    #   user = models.ForeignKey(User)
    curp = models.CharField(max_length=30)
     # domicilio
    piso = models.IntegerField()
     # type: Number,
    departamento = models.IntegerField()
     # type: Number,
    telefono = models.CharField(max_length=10)
     # type: String, models.IntegerField()
    correo = models.EmailField()
    #  type: String,
    fecha_nac = models.DateField()
    fecha_creado = models.DateTimeField(auto_now_add=True)
    fecha_actualizado = models.DateTimeField(auto_now=True)
    # Campos con opciones
    CATALOGO_PERFIL = [
        ('1', 'Usuario Inquilino'),
        ('2', 'Usuario Visitante'),
        ('3', 'Usuario Trabajador'),
    ]
    id_perfil = models.CharField(max_length=2, choices=CATALOGO_PERFIL, default='2')
    # Campos con opciones
    CATALOGO_STATUS = [
        ('1', 'ALTA'),
        ('2', 'BAJA'),
    ]
    id_status = models.CharField(max_length=2, choices=CATALOGO_STATUS, default='3')

    def __str__(self):
        # return "{} {} {}". format(self.nombre, self.ap_paterno, self.ap_materno)
        return f'Id {self.id_usuario} -> {self.nombre} {self.ap_paterno} {self.ap_materno}'


class Sesion(models.Model):
    id_sesion = models.AutoField(primary_key=True)
    # Relacion 1 a muchos, 
    # donde 1 usuario puede tener muchas sesiones 
    # Una sesion solo le pertenece a un usuario
    # Una sesion forzosamente deb permanecer a un usuario
    # por lo que no debe ser valor nulo null=False
    # ni almacenar valores vacios blank=False
    id_usuario = models.ForeignKey(Usuario, 
                                   on_delete=models.CASCADE,
                                   null=False,
                                   blank=False)
    CATALOGO_TIPO_SESION = [
        ('1', 'FACIAL'),
        ('2', 'VOZ'),
        ('3', 'HUELLA'),
    ]
    id_tipo_sesion = models.CharField(max_length=2, choices=CATALOGO_TIPO_SESION)

    # id_tipo_sesion = models.IntegerField(null=False, blank=False, choices=CATALOGO_TIPO_SESION, default='4')
    completado = models.BooleanField(default=False)
    # CATALOGO_STATUS_SESION = [
    #     ('0', 'NO REGISTRADO'),
    #     ('1', 'REGISTRADO'),
    # ]
    # id_status_sesion = models.CharField(max_length=2, choices=CATALOGO_STATUS_SESION, default='0')
    dato = models.BinaryField(editable = True)
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    fecha_actualizacion = models.DateTimeField(auto_now=True)
      
    def __str__(self):
         return f'{self.id_tipo_sesion}'





# class Usuario_Tipo_Sesion(models.Model):
#     id_usuario = models.ForeignKey(Usuario, on_delete=models.CASCADE)
#     id_tipo_sesion = models.ForeignKey(Usuario_Sesion, on_delete=models.CASCADE)
#     image = models.BinaryField()

#contraseña generada por default[5dias para validar el alta sino se dara de baja]
#ingresa para los metodos de inicio de sesion
#si hay datos en la bd, bienvenidos ingresa tus datos boton siguiente, 
# imagen, huella(en el sensor),voz(frase), guardado entra, los datos que esten completados[palomita]
#omitir en algun tipo de sesion, huella y voz (opcional en registro mas tarde)
# display none, administrador sin editar solo visualiza y( elimina registro de sesion, solo desbloquea el bloqueo ) ocultar botones
# inquilinos vista cambiar por tabla, perfil por usuario_(Detail)
#tabla interna de desbloqueo o bloqueo y actualizacio fexha e id de quien lo hizo
#tabla is sesion de id de usuario fecha de alta, actualizacion id de quien actualizo
# Tbl_tipo_sesion
# Idsesion
# Idusuario
# Idtiposesion
# Dato [huella, foto, audio, contraseña] clob o blob
# Fecha_alta
# Fecha_modificacion
# IdUsuario_modificacion

#  IdDato ( numero de muestra)


