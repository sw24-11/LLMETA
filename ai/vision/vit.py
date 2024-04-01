from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import requests

def classify_image_with_vit(image_path):
    model_name = 'google/vit-base-patch16-224'
    model = ViTForImageClassification.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    if image_path.startswith('http'):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label.get(predicted_class_idx, f"Label index {predicted_class_idx}")

    return predicted_class

image_path = 'C:/Users/kbh/img/cropped_person_3.jpg'  
predicted_label = classify_image_with_vit(image_path)
print(f'Predicted label: {predicted_label}')
