import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from helpers import *
from kinect_variety_fusion_2d import VISUALIZE
from piv_artificial_data_test import IMAGE_SIZE
from visualization import plot_F_squared_around_x
from scipy.optimize import minimize, Bounds, least_squares
from time import time
from fp_piv import *

VISUALIZE = True
DATA_PATH = "data/artificial_data/face_ply/PLY_Pasha.ply"

IMAGE_SIZE = 400

def load_mesh_into_xyz_pc():
    # mesh = o3d.io.read_triangle_mesh(DATA_PATH)
    # pcd = mesh.sample_points_poisson_disk(number_of_points=2)
    pcd = o3d.io.read_point_cloud(DATA_PATH)
    pcd = pcd.voxel_down_sample(voxel_size=20)
    if VISUALIZE:
        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.add_geometry(pcd)
        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([0.8, 0.8, 0.8])
        viewer.run()
        viewer.destroy_window()
    xyz = pcd2xyz(pcd)
    return xyz.T

def project_points(input_points, rot, t, K):
    img_points, _ = cv2.projectPoints(input_points, rot, t, K, None)
    img = np.ones([IMAGE_SIZE,IMAGE_SIZE,3], dtype=np.uint8)*220 
    for point in img_points:
        img[int(point[0][1]), int(point[0][0]), :] = [0,0,0]

    return [point for pt in img_points for point in pt], img

def main():
    points = load_mesh_into_xyz_pc()

    K1 = np.array([[.5, 0., IMAGE_SIZE/2],
                    [0., .5, IMAGE_SIZE/2],
                    [0., 0., 1.]])
    Rx = np.array([[1., 0., 0.],
                    [0., 0., -1.],
                    [0., 1., 0.]])
    Ry = np.array([[0., 0., 1.],
                    [0., 1., 0.],
                    [-1., 0., 0.]])
    Rz = np.array([[0., -1., 0.],
                    [1., 0., 0.],
                    [0., 0., 1.]])
    t1 = np.array([0., -100., -100.])
    t2 = np.array([0., 0., -6.])
    t3 = np.array([0., 0., -8.])
    img1_points, img1 = project_points(points, np.eye(3), t1, K1)
    img2_points, img2 = project_points(points, np.eye(3), t2, K1)
    img3_points, img3 = project_points(points, np.eye(3), t3, K1)

    fig = plt.figure("Views")
    ax = fig.add_subplot(2, 3, 1)
    imgplot = plt.imshow(img1)
    ax.set_title('View 1')
    ax = fig.add_subplot(2, 3, 2)
    imgplot = plt.imshow(img2)
    ax.set_title('View 2')
    ax = fig.add_subplot(2, 3, 3)
    imgplot = plt.imshow(img3)
    ax.set_title('View 3')
    # ax = fig.add_subplot(2, 3, 4)
    # imgplot = plt.imshow(img4)
    # ax.set_title('View 4')
    # ax = fig.add_subplot(2, 3, 5)
    # imgplot = plt.imshow(img_new)
    # ax.set_title("Ground truth")
    # ax = fig.add_subplot(2, 3, 6)
    # imgplot = plt.imshow(novel_view)
    # ax.set_title("Novel view")
    plt.show()

if __name__ == "__main__":
    main()