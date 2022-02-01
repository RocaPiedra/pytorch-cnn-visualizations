"""

@author: Pablo Roca - github.com/RocaPiedra
"""
import os
import copy
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map

import torch
from torch.autograd import Variable
from torchvision import models
from misc_functions import apply_colormap_on_image, save_image

def get_image_path(path, filename):
    if filename == None:
        onlyimages = [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) & f.endswith(('.jpg','.png'))]
        return onlyimages
    else:
        image_path = path + filename
        return image_path

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

# def get_image(path):
#     if path.endswith('.jpg' or '.png'):
#         image = Image.open(path).convert('RGB')
#     elif path.endswith('.avi'):
