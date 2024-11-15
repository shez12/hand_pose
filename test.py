import sys

from  dino_vit_features import keypoint_utils
import mediapy as media
import numpy as np
import matplotlib.pyplot as plt
import cv2


# image_array1 = cv2.imread("dino_vit_features/images/arm_1.jpg")
# image_array2 = cv2.imread("dino_vit_features/images/arm_2.jpg")

image_array1 = cv2.imread("image/camera1_rgb_0.jpg")
image_array2= cv2.imread("image/camera1_rgb_1.jpg")


image_array3 = cv2.imread("data/images-20241111-113054/camera1_rgb_0.jpg")
print(image_array3.shape)

for image_i in range(0,2):
  image_path = f"image/camera1_rgb_{image_i}.jpg"
  image = media.read_image(image_path)
  image = media.resize_image(image, (240,320))
  print("resized image shape",image.shape)
  media.write_image(f"image/camera1_rgb_{image_i}.jpg", image)



# Select the paths of a pair of images from which you want to extract keypoint correspondences.
patches_xy, desc1, descriptor_vectors, num_patches = keypoint_utils.extract_descriptors(image_array1, image_array2, num_pairs = 5, load_size = 240)




y_list, x_list = [], []

colors = np.random.randint(0,255, 3*20).reshape((-1,3))
for image_i in range(0,2):
  image_array = cv2.imread(f"image/camera1_rgb_{image_i}.jpg")
  image_arrays = [image_array]
  # image_path = f"dino_vit_features/images/arm_{image_i}.jpg"
  map3, _ = keypoint_utils.extract_desc_maps(image_arrays)
  print(map3[0].shape)
  cs_ys_list, cs_xs_list = keypoint_utils.extract_descriptor_nn(descriptor_vectors, map3[0], num_patches, False)
  y_list.append(cs_ys_list)
  x_list.append(cs_xs_list)
  # input_image = media.resize_image(media.read_image(image_array), (480,640))
  output_image = keypoint_utils.draw_keypoints(image_array, cs_ys_list, cs_xs_list, colors = colors)
  plt.imshow(output_image)
  plt.show()
