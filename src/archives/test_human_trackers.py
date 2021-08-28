import cv2
from matplotlib import pyplot as plt
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

fig = plt.figure("Result")

# For static images:
IMAGE_FILES = ["data/rgb_20210421-155829.jpg", "data/rgb_20210421-155835.jpg"]
IMAGE_FILES = ["data/Tx55cm/rgb_0.jpg", "data/Tx55cm/rgb_1.jpg"]
DEPTH_IMAGES = ["data/Tx55cm/depth_0.npy", "data/Tx55cm/depth_1.npy"]
with mp_holistic.Holistic(
     min_detection_confidence=0.7,
    static_image_mode=True) as holistic:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape
    depth = np.load(DEPTH_IMAGES[idx])/3000
    depth_image = cv2.merge((depth,depth,depth))
    # Convert the BGR image to RGB before processing.
    results = holistic.process(image)

    if results.pose_landmarks:
      print(
          f'Nose coordinates: ('
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
          f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
      )
    # Draw pose, left and right hands, and face landmarks on the image.
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(
        annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # mp_drawing.draw_landmarks(
    #     annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(
        depth_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)

    ax = fig.add_subplot(2, 2, 2*idx+1)
    imgplot = plt.imshow(annotated_image)
    ax.set_title('Image {0}'.format(idx))
    ax = fig.add_subplot(2, 2, 2*idx+2)
    imgplot = plt.imshow(depth_image)
    ax.set_title('Depth image {0}'.format(idx))
    # Plot pose world landmarks.
    # mp_drawing.plot_landmarks(
    #     results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
plt.show()