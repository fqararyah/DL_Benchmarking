import sys
import os
import numpy

from torch import imag

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
from data_processing import PreprocessYOLO, PostprocessYOLO, ALL_CATEGORIES

input_resolution_yolov4_HW = (608, 608)
preprocessor = PreprocessYOLO(input_resolution_yolov4_HW)

image_raw, image = preprocessor.process('/home/fareed/wd/vedliot/D3.3_Accuracy_Evaluation/coco/coco_val2017/000000000885.jpg')

bgr = numpy.flip(image, 1)

print(image.shape)
# print(image[0][0][20:30])
# print('**********************')
# print(bgr[0][0][20:30])
# print('*****************************************************')
# print(image[0][1][20:30])
# print('**********************')
# print(bgr[0][1][20:30])
# print('*****************************************************')
# print(image[0][2][20:30])
# print('**********************')
# print(bgr[0][2][20:30])

