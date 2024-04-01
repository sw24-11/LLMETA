import torch
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
import os
from torchvision import transforms
from torchvision.models import resnet152, ResNet152_Weights


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
img_path = 'C:/Users/kbh/img/Dog_Breeds.jpg'  
original_image = Image.open(img_path)

def person_processing(box):
    #boxes = results.xyxy[0][results.xyxy[0][:, -1] == 0]
    #for i, box in enumerate(boxes):
    x_min, y_min, x_max, y_max, _, _ = box.cpu().numpy()
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))

    cropped_image_cv = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
    
    try:
        analysis_results = DeepFace.analyze(img_path=cropped_image_cv, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
        analysis = analysis_results[0] if analysis_results else {}
        
        print(f"Analysis for person {i+1}:")
        print(f"Age: {analysis.get('age', 'N/A')}")
        print(f"Gender: {analysis.get('dominant_gender', 'N/A')}")
        print(f"Race: {analysis.get('dominant_race', 'N/A')}")
        print(f"Emotion: {analysis.get('dominant_emotion', 'N/A')}")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

def NonCommon_processing(box):
    #model = models.resnet50(pretrained=True)
    x_min, y_min, x_max, y_max, _, _ = box.cpu().numpy()
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
    #cropped_image_cv = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)

    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    #image = Image.open(image_path)
    img_tensor = preprocess(cropped_image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
    
    _, predicted_class = outputs.max(1)
    predicted_class = predicted_class.item()

    with open("C:/Users/kbh/Code/project2/vision/imagenet_classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes[predicted_class]

results = model(img_path)
results.print()
label_index = results.xyxy[0][:, -1]
int_list = [int(item) for item in label_index.tolist()]

common_labels = {'orange': 49,'refrigerator': 72,'broccoli': 50,'microwave': 68,'cup': 41,
                'zebra': 22, 'mouse': 64,'backpack': 24,'banana': 46,'umbrella': 25,'toaster': 70,
                'traffic light': 9,'parking meter': 12,'pizza': 53,'vase': 75,'laptop': 63,'kite': 33}

#extracted_set = set(int_list)
common_labels_list = list(common_labels.values())
#intersection_set = extracted_set.intersection(common_labels_list)

for i, box in enumerate(results.xyxy[0]):
    x_min, y_min, x_max, y_max, confidence, class_id = box.cpu().numpy()
    if class_id==0:
        person_processing(box)
    elif class_id in common_labels_list:
        print("common")
    else:
        print(NonCommon_processing(box))
        


# for label in int_list:
#     if label==0:
#         print("person")
#     elif label in common_labels_list:
#         print("common")
#     else:
#         print("Not common")



        
