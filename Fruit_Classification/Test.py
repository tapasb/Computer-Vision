import os
import numpy as np
from PIL import Image, ImageDraw
from tensorflow import keras
from ultralytics import YOLO

# 1. Load the trained fruit classification model
model_classification = keras.models.load_model("fruit_classification_model.h5")  # Path to your model
class_names = sorted(os.listdir("data/train"))  # Path to your training data

# 2. Load a pre-trained object detection model (YOLOv5 in this example)
model_detection = YOLO('yolov5s.pt')  # You can use yolov5m, yolov5l, or other versions

def detect_and_classify(image_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    results = model_detection(image_path)

    for result in results:  # Iterate through the list of Results objects
        boxes = result.boxes  # Access the boxes attribute of each Result

        for *xyxy, conf, cls in boxes.data:
            if conf > 0.5:  # Confidence threshold (adjust as needed)
                x1, y1, x2, y2 = map(int, xyxy)

                try:
                    fruit_image = img.crop((x1, y1, x2, y2))
                    fruit_image = fruit_image.resize((100, 100))  # Resize to match classification model input
                    fruit_image_array = np.array(fruit_image) / 255.0
                    fruit_image_array = np.expand_dims(fruit_image_array, axis=0)

                    predictions = model_classification.predict(fruit_image_array)
                    predicted_class_index = np.argmax(predictions)
                    predicted_class_name = class_names[predicted_class_index]

                    draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
                    draw.text((x1, y1 - 10), f"{predicted_class_name} ({conf:.2f})", fill="red")

                except ValueError:
                    print(f"Error processing region: {x1, y1, x2, y2}. Skipping.")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}. Skipping.")

    return img

# Example usage:
image_path = "TestImage.jpg"
result_image = detect_and_classify(image_path)
result_image.show()
result_image.save("result_image_with_detections_and_classification.jpg")