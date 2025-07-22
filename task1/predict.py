import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2


model = load_model("keras_model.h5")

with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

img = cv2.imread("test.jpg")  
img = cv2.resize(img, (224, 224))  
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

prediction = model.predict(img)
predicted_class_index = np.argmax(prediction)
predicted_class_name = class_names[predicted_class_index]

print("Predicted Class:", predicted_class_name)

 
img_for_display = (img[0] * 255).astype(np.uint8)
cv2.putText(img_for_display, predicted_class_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imwrite("output.jpg", img_for_display)
print(" The output image is saved as output.jpg")

