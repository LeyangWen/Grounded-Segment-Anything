import shutil
import os
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

image_folder = r'/media/leyang/New Volume/i3ce2023_datathon/bing_images/images'
good_image_folder = r'/media/leyang/New Volume/i3ce2023_datathon/bing_images/good_images'
for folder_idx, (root, dirs, files) in enumerate(os.walk(image_folder)):
    # search through the folder and subfolders for all images
    # pass if file not jpg or jpeg
    if folder_idx < 40:
        continue
    for file_idx, file in enumerate(files):
        if file.endswith('.jpg') or file.endswith('.jpeg'):
            pass
        else:
            continue
        # if file resolution is smaller than (1000,500)
        image_name = os.path.join(root, file)
        good_image_name = image_name.replace('.jpg', '.jpeg').replace(image_folder, good_image_folder)
        image = Image.open(image_name)
        if image.size[0] < 1000 or image.size[1] < 500:
            continue
        # if image have a lot of white pixels
        image = image.convert('RGB')
        image_np = np.array(image)
        white_pixel_R = image_np[:, :, 0] > 240
        white_pixel_G = image_np[:, :, 1] > 240
        white_pixel_B = image_np[:, :, 2] > 240
        white_pixel = white_pixel_R & white_pixel_G & white_pixel_B
        if np.sum(white_pixel) > 0.35 * image.size[0] * image.size[1]:
            continue

        # copy to good image folder
        if not os.path.exists(os.path.dirname(good_image_name)):
            os.makedirs(os.path.dirname(good_image_name))
        shutil.copy(image_name, good_image_name)
        print(f'{folder_idx} copy {image_name} to {good_image_name}')