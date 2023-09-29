import cv2
from yolo_predictions import YOLO_Pred
import os

yolo = YOLO_Pred('./Model/weights/best.onnx','data.yaml')

img = cv2.imread('image.jpg')


img_pred = yolo.predictions(img)

cv2.imwrite(os.path.join('photos', "1.jpg"), img_pred)

cv2.imshow('prediction image', img_pred)
cv2.waitKey(0)
cv2.destroyAllWindows()