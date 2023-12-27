
# Register your models here.
from django.contrib import admin
from .models import Usuario, Sesion
# Register your models here.

# class UsuarioAdmin(admin.ModelAdmin):
#   readonly_fields = ('creado', )
class UsuarioAdmin(admin.ModelAdmin):
    readonly_fields = ('fecha_creado','fecha_actualizado' )
    list_display = ["id_usuario","nombre", "ap_paterno", "ap_materno","piso", "departamento", 'id_perfil']

class SesionAdmin(admin.ModelAdmin):
    readonly_fields = ('fecha_creacion','fecha_actualizacion' )
    list_display = ['id_sesion', 'id_usuario', 'id_tipo_sesion','dato', 'fecha_creacion']

admin.site.register(Usuario, UsuarioAdmin)
admin.site.register(Sesion,SesionAdmin)

