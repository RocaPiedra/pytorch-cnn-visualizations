"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from time import sleep
from PIL import Image
from cv2 import *
import numpy as np
import torch

from misc_functions import get_example_params, save_class_activation_images, get_image_path, preprocess_image


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        if self.model.__class__.__name__ == 'ResNet':
            for module_pos, module in self.model._modules.items():
                if module_pos == "avgpool":
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
                    x = module(x)
                    print("hook layer for ResNet: ", module)
                else:
                    x = module(x)
                    print("forward pass in layer for ResNet: ", module)
        else:
            for module_pos, module in self.model.features._modules.items():
                x = module(x)  # Forward
                if int(module_pos) == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
                    print("hook layer: ", module)
                else:
                    print("forward pass in layer: ", module)
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the ResNet model https://github.com/utkuozbulak/pytorch-cnn-visualizations/issues/50
        if self.model.__class__.__name__ == 'ResNet':
            # Forward pass on the convolutions
            conv_output, x = self.forward_pass_on_convolutions(x)
            x = x.view(x.size(0), -1)  # Flatten
            # Forward pass on the classifier
            try:
                print('\n************\nResNet uses .fc \n************\n')
                x = self.model.fc(x)
            except:
                print('\n************\nResNet .fc failed \n************\n')
                x = self.model.classifier(x)
        else:
            # Forward pass on the convolutions
            conv_output, x = self.forward_pass_on_convolutions(x)
            x = x.view(x.size(0), -1)  # Flatten
            # Forward pass on the classifier
            x = self.model.classifier(x)
        return conv_output, x

    
    # def resnet_forward_pass_on_convolutions(self, x):
    #     """
    #         Does a forward pass on convolutions, hooks the function at given layer
    #         https://github.com/utkuozbulak/pytorch-cnn-visualizations/issues/50
    #     """
    #     conv_output = None
    #     for module_pos, module in self.model._modules.items():
    #         if module_pos == "avgpool":
    #             x.register_hook(self.save_gradient)
    #             conv_output = x  # Save the convolution output on that layer
    #             x = module(x)
    #             print("hook layer for ResNet: ", module)
    #         else:
    #             x = module(x)
    #             print("forward pass in layer for ResNet: ", module)
    #     return conv_output, x

    # def resnet_forward_pass(self, x):
    #     """
    #         Does a full forward pass on the ResNet model
    #         https://github.com/utkuozbulak/pytorch-cnn-visualizations/issues/50
    #     """
    #     if self.model.__class__.__name__ == 'ResNet':
    #         # Forward pass on the convolutions
    #         conv_output, x = self.resnet_forward_pass_on_convolutions(x)
    #         x = x.view(x.size(0), -1)  # Flatten
    #         # Forward pass on the classifier
    #         try:
    #             x = self.model.fc(x)
    #             print('ResNet uses .fc \n')
    #         except:
    #             x = self.model.classifier(x)
    #             print('ResNet .fc failed \n')
    #     return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Have a look at issue #11 to check why the above is np.ones and not np.zeros
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam


if __name__ == '__main__':
    # Get params
    target_example = 4  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    print()
    path = '../input_images/carla_input/'
    image_paths = get_image_path(path,None)
    # print(image_paths)
    # sleep(2)
    # pretrained_model.eval()
    # for image_path in image_paths:
    #     input_image = Image.open(image_path).convert('RGB')
    #     input_image.show()
    #     sleep(1)
    #     preprocessed_image = preprocess_image(input_image, resize_im = True)
    #     if torch.cuda.is_available():
    #         preprocessed_image = preprocessed_image.to('cuda')
    #         pretrained_model.to('cuda')

    #     with torch.no_grad():
    #         output = pretrained_model(preprocessed_image)
    #     print('Output of the model for image ',image_path, 'is: \n',output)
    
    # print('Outputs:',output[0],'\n',output.size())
    # probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print('After SoftMax: ',probabilities[0],'\n',probabilities.size())

    # Grad cam
    grad_cam = GradCam(pretrained_model)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Grad cam completed')
