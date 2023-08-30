
from PIL import Image
import numpy as np

path = '/home/fareed/wd/vedliot/D3.3_Accuracy_Evaluation/imagenet/images/ILSVRC2012_val_00009878.JPEG'

image = Image.open(path)


#image.convert('RGB')

image.show()

narr = np.asarray(image)

print(narr.shape)

print(narr.ndim)

narr = np.repeat(narr[:, :, np.newaxis], 3, axis=2)

print(narr.shape)

print(narr.ndim)

print(image.size)

path = '/home/fareed/wd/vedliot/D3.3_Accuracy_Evaluation/imagenet/images/ILSVRC2012_val_00000001.JPEG'

image = Image.open(path)

print(image.size)

narr = np.asarray(image)

print(narr.shape)