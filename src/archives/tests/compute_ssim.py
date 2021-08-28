import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage import img_as_float
from skimage.metrics import structural_similarity as ssim

# img = cv2.imread("src/tests/human_vertB_textMap_landmarks.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("src/tests/textMap_human4K_vertB_landmarks.png", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("src/tests/textMap_human4K_landmarks.png", cv2.IMREAD_GRAYSCALE)

fig_img = plt.figure("Input")
plt.imshow(img, cmap='gray')

# gt = img_as_float(img[251:536, 135:335])
# synth = img_as_float(img[245:530, 920:1120])
# gt = img_as_float(img[170:475, 299:454])
# synth = img_as_float(img[165:470, 1080:1235])
gt = img_as_float(img[220:350, 288:388])
synth = img_as_float(img[221:351, 1070:1170])

ssim_val = ssim(gt, synth, data_range=synth.max()-synth.min())
print(ssim_val)

fig = plt.figure("Texture mapping result")
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(gt, cmap='gray')
ax.set_title('Ground truth view')
ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(synth, cmap='gray')
ax.set_title('Virtual image')
plt.show()


