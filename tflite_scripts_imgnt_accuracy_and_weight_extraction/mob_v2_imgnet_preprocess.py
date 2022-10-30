import tensorflow.keras.preprocessing.image as img_proc
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array


import os

DATA_PATH = '/media/SSD2TB/shared/vedliot_evaluation/D3.3_Accuracy_Evaluation/imagenet/imagenet_val2012'

out_path = './preprocessed_tst_images/'

image_names = []
test_images = []
def locate_images(path):
    
    # dirs=directories
    for (root, dirs, file) in os.walk(path):
        for f in file:
            if 'ILSVRC2012_val_00003599.JPEG' in f:
                test_images.append(os.path.abspath(os.path.join(path, f)))
                image_names.append(f)
                #print(image_list[-1])

locate_images(DATA_PATH)

def mob_v2_first_quantization(arr):
    #this is a simplificatio of:
    # a- 0.007843137718737125 * (q - 127)
    # b- 0.007843137718737125 * (q + 1)
    return arr - 128 

limit = min(len(test_images), 1000)
for i in range(limit):
    a_test_image = img_proc.load_img(test_images[i], target_size = (224, 224))
    numpy_image = img_to_array(a_test_image, dtype = np.uint8)
    numpy_image = np.transpose(numpy_image, (2, 0, 1))
    image_batch = np.expand_dims(numpy_image, axis = 0)
    image_batch = np.reshape(image_batch, (image_batch.size))
    #print(image_batch[-100:])
    image_batch = mob_v2_first_quantization(image_batch)
    np.savetxt(out_path + image_names[i].split('.')[0] + '.txt',image_batch.astype(np.int8), fmt='%i')
    #print(processed_image.shape)