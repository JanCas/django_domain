from django.contrib import admin

from .models.ModelParams import ModelParams
from .models.Model import Model
from .models.Metrics import Metrics
from .models.Callbacks import Callbacks
from .models.ModelScores import ModelScores


# Register your models here.

# inline models
class ModelParamsInLine(admin.TabularInline):
    model = ModelParams
    max_num = 1


class MetricsInLine(admin.TabularInline):
    model = Metrics
    max_num = 2


class CallbacksInLine(admin.TabularInline):
    model = Callbacks
    max_num = 2

class ModelScoresInLine(admin.TabularInline):
    model = ModelScores
    max_num = 0

# head models
class ModelAdmin(admin.ModelAdmin):
    inlines = (ModelParamsInLine, MetricsInLine, CallbacksInLine, ModelScoresInLine)

    list_display = ('name', 'cbfv', 'mat_prop', 'batch_size', 'epochs', 'trained')

    actions = ['train']

    def train(self, request, queryset):
        for model in queryset:
            model.train()


admin.site.register(Model, ModelAdmin)
