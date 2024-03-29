import torch
from matplotlib import pyplot as plt
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
img_path = 'C:/Users/kbh/img/human1.jpg'
original_image = Image.open(img_path)

results = model(img_path)
results.print()  
detected_classes = results.xyxy[0][:, -1].cpu().numpy()

boxes = results.xyxy[0][results.xyxy[0][:, -1] == 0]

for i, box in enumerate(boxes):
    x_min, y_min, x_max, y_max, _, _ = box.cpu().numpy()
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max)) 

    save_path = f'C:/Users/kbh/img/cropped_person_{i+1}.jpg'
    cropped_image.save(save_path)
    print(f"Cropped person {i+1} saved to {save_path}")


#breeds
#race
#or.. BLIP?