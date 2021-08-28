import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread("data/human4K/rgb_1.jpg", cv2.IMREAD_COLOR)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = img1[1100:2200, 1300:2400]
img2 = cv2.imread("data/human4K/rgb_2.jpg", cv2.IMREAD_COLOR)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = img2[1100:2200, 1300:2400]
img3 = cv2.imread("data/human4K/rgb_3.jpg", cv2.IMREAD_COLOR)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img3 = img3[1100:2200, 1300:2400]


detector = cv2.ORB_create(15000)
descriptor = cv2.xfeatures2d.BEBLID_create(.75, 101)

kp1 = (detector.detect(img1, None))
kp1, des1 = descriptor.compute(img1, kp1)
kp2 = (detector.detect(img2, None))
kp2, des2 = descriptor.compute(img2, kp2)
kp3 = (detector.detect(img3, None))
kp3, des3 = descriptor.compute(img3, kp3)

FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                table_number = 6, # 12
                key_size = 12,     # 20
                multi_probe_level = 2) #2
search_params = dict(checks=50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)

matches = matcher.knnMatch(des1, des2, k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.66*n.distance:
        good.append([m])
print(len(good))

# cv.drawMatchesKnn expects list of lists as matches.
img_out = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_DEFAULT)
# img4 = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
# img5 = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)
fig1 = plt.figure("Matches")
plt.imshow(img_out)
# fig2 = plt.figure("Keypoints image 1")
# plt.imshow(img4)
# fig3 = plt.figure("Keypoints image 2")
# plt.imshow(img5)
plt.show()