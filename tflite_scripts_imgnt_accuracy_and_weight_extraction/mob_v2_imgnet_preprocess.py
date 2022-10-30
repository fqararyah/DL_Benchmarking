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
            if '.JPEG' in f:
                test_images.append(os.path.abspath(os.path.join(path, f)))
                image_names.append(f)
                #print(image_list[-1])

locate_images(DATA_PATH)


limit = 1000
for i in range(limit):
    a_test_image = img_proc.load_img(test_images[i], target_size = (224, 224))
    numpy_image = img_to_array(a_test_image, dtype = np.uint8)
    image_batch = np.expand_dims(numpy_image, axis = 0)
    image_batch = np.reshape(image_batch, (image_batch.size))
    np.savetxt(out_path + image_names[i].split('.')[0] + '.txt',image_batch, fmt='%i')
    #print(processed_image.shape)