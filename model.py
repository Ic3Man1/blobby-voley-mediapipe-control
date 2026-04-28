import tensorflow as tf
import numpy as np
import cv2

class GestureClassifier:
    def __init__(self, model_path, class_names):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        self.img_size = (224, 224)

    def predict(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img_array = np.expand_dims(img, axis=0)
        
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        predictions = self.model.predict(img_array, verbose=0)
        
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        return self.class_names[class_idx], confidence