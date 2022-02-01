import torch
from torchvision import models

def choose_model(modelname = None):
    if modelname == 'resnet':
        model = models.resnet18(pretrained=True)
    elif modelname == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        option = int(input('Model is not defined choose from the available list:\n'
        '1. Alexnet\n2. ResNet\n'))
        if option == 1:
            model = models.alexnet(pretrained=True)
            print('Alexnet is the chosen classifier')
        elif option == 2:
            model = models.resnet50(pretrained=True)
            print('ResNet is the chosen classifier')
        else:
            print('Option incorrect, set default model: Alexnet')
            model = models.alexnet(pretrained=True)

    return model

