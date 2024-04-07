import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace

def person_processing(box, original_image, i):
    #boxes = results.xyxy[0][results.xyxy[0][:, -1] == 0]
    #for i, box in enumerate(boxes):
    #original_image = Image.open(img_pth)

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

            