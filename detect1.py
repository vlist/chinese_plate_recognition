import tensorflow as tf
from tensorflow import keras as K
import cv2
import numpy as np
from cnn import cnn_predict
model_path = "./models/cnn_model.h5"

model = K.models.load_model(model_path)
img_path = "plate_dataset/test/川A679DE-0237212643678-90_87-212&509_488&607-481&600_215&605_217&497_483&492-22_0_30_31_33_3_4-165-50.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
lic = []
lic.append(img)
predict = cnn_predict(model,lic)

print(predict[-1][-1])