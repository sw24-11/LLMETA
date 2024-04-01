import torch
from torchvision import transforms
from torchvision.models import resnet152, ResNet152_Weights
from PIL import Image

def classify_image(image_path):
    #model = models.resnet50(pretrained=True)
    model = resnet152(weights=ResNet152_Weights.DEFAULT)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    img_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
    
    _, predicted_class = outputs.max(1)
    predicted_class = predicted_class.item()

    with open("C:/Users/kbh/Code/project2/vision/imagenet_classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes[predicted_class]

image_path = 'C:/Users/kbh/img/cropped_person_3.jpg' 
predicted_label = classify_image(image_path)
print(f'Predicted label: {predicted_label}')
