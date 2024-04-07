import torch
import numpy as np
from PIL import Image
import os
import cv2
from deepface import DeepFace

from second_classification.vit import NonCommon_processing_ViT
from second_classification.resnet import NonCommon_processing_Resnet
from second_classification.person_processing import person_processing
from intersection_imagenet import common_labels
from blip import open_vocabulary_classification_blip
from transformers import BlipProcessor, BlipForConditionalGeneration

rank = 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
img_path = 'C:/Users/kbh/img/3.jpg'  #"C:\Users\kbh\img\image-8.png"
original_image = Image.open(img_path)

results = model(img_path)
results.print()
label_index = results.xyxy[0][:, -1]
int_list = [int(item) for item in label_index.tolist()]

common_labels_list = list(common_labels.values())
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(rank)

open_vocabulary_classification_blip(raw_image=original_image, blip_processor=blip_processor, blip_model=blip_model,rank=rank)

#extracted_set = set(int_list)
#intersection_set = extracted_set.intersection(common_labels_list)

for i, box in enumerate(results.xyxy[0]):
    x_min, y_min, x_max, y_max, confidence, class_id = box.cpu().numpy()
    if class_id==0:
        person_processing(box=box, original_image=original_image, i=i)
    elif int(class_id) in common_labels_list:
        print("common")
    else:
        print(NonCommon_processing_Resnet(box=box, original_image=original_image))


 

        