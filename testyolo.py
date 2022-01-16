#!/usr/bin/env conda run -n yolov5 python
import torch
import PIL
import numpy as np
import pandas

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
imgs = ['https://ultralytics.com/images/zidane.jpg','./input_images/carla_input/1.png','./input_images/carla_input/2.png']  # batch of images

# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)