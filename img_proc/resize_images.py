import cv2
import os

imaged_dir = '/media/SSD2TB/shared/vedliot_evaluation/D3.3_Accuracy_Evaluation/imagenet/imagenet_val2012/'
resized_imaged_dir = '/media/SSD2TB/shared/vedliot_evaluation/D3.3_Accuracy_Evaluation/imagenet/imagenet_val2012_resized/'

dst_width = 224
dst_height = 224
dst_size = (dst_width, dst_height)

for filename in os.listdir(imaged_dir):
    f = os.path.join(imaged_dir, filename)
    # checking if it is a file
    print(f)
    if os.path.isfile(f) and '.jpeg' in filename.lower():
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        
        resized_down_img = cv2.resize(img, dst_size)
        cv2.imwrite(os.path.join(resized_imaged_dir, filename), resized_down_img)

#img = cv2.imread(img_file, cv2.IMREAD_COLOR)