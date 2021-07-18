import cv2
from matplotlib import pyplot as plt
import numpy as np

from corresponding_points_selector import CorrespondingPointsSelector
from parameterized_image_variety import ParameterizedImageVariety

FOLDER_NAME = "data/artificial_data/74/"
TIMESTAMP = "_20210629-1539"

def load_rgbd2(idx):
    rgb_cam = cv2.imread(FOLDER_NAME+"photo/"+idx+".jpg", cv2.IMREAD_COLOR)
    rgb_cam = cv2.cvtColor(rgb_cam, cv2.COLOR_BGR2RGB)
    depth_cam = cv2.imread(FOLDER_NAME+"depth/"+idx+".png", cv2.IMREAD_UNCHANGED)

    return rgb_cam, depth_cam

def load_rgbd(idx):
    rgb_cam = cv2.imread(FOLDER_NAME+"rgb_"+idx+TIMESTAMP+".jpg", cv2.IMREAD_COLOR)
    rgb_cam = cv2.cvtColor(rgb_cam, cv2.COLOR_BGR2RGB)
    depth_cam = np.load(FOLDER_NAME+"depth_"+idx+TIMESTAMP+".npy")

    return rgb_cam, depth_cam


def normalize_uvd(q0, d0, q1, d1, q2, d2, pts, d_pts, max_px, max_d):
    nb_views = len(q0)

    q0_n = []
    d0_n = []
    q1_n = []
    d1_n = []
    q2_n = []
    d2_n = []
    pts_n = [[] for i in range(nb_views)]
    d_pts_n = [[] for i in range(nb_views)]

    for i in range(nb_views):
        q0_n.append((q0[i][0]/max_px, q0[i][1]/max_d))
        d0_n.append(d0[i]/max_d)
        q1_n.append((q1[i][0]/max_px, q1[i][1]/max_px))
        d1_n.append(d1[i]/max_d)
        q2_n.append((q2[i][0]/max_px, q2[i][1]/max_px))
        d2_n.append(d2[i]/max_d)
        for j, pt in enumerate(pts[i]):
            pts_n[i].append((pt[0]/max_px, pt[1]/max_px))
            d_pts_n[i].append(d_pts[i][j]/max_d)

    return q0_n, d0_n, q1_n, d1_n, q2_n, d2_n, pts_n, d_pts_n

def main():

    rgb_cams = []
    depth_cams = []
    cams_idx = range(600,800,25)
    for idx in cams_idx:
        # if i != 2: # keep view 2 for reconstruction
        rgb_cam, depth_cam = load_rgbd2(str(idx))
        rgb_cams.append(rgb_cam)
        depth_cams.append(depth_cam)
        #TODO: get the maximum depth value to set upper bound on solution space
    
    frame_height, frame_width, _ = np.shape(rgb_cams[0])
    max_depth = np.max(depth_cams)

    ptSelector = CorrespondingPointsSelector()

    q0, d0, q1, d1, q2, d2, pts, d_pts = ptSelector.points_selection(rgb_cams, depth_cams)

    pts_px = pts.copy()
    # Normalize the pixel position and depth to be on a [0,1] scale for a robust optimization
    q0, d0, q1, d1, q2, d2, pts, d_pts = normalize_uvd(q0, d0, q1, d1, q2, d2, pts, d_pts, max_px=frame_width, max_d=max_depth)

    virtual_cam = 675
    virtual_view = cams_idx.index(virtual_cam)

    piv = ParameterizedImageVariety(q0, d0, q1, d1, q2, d2, pts, d_pts, virtual_view, frame_width, frame_height, max_depth)

    virtual_pts, virtual_d_pts = piv.get_virtual_pts()


    for idx in range(len(pts[0])):
        rgb = np.random.rand(3,)*255
        for i in range(len(rgb_cams)):
            rgb_cams[i] = cv2.circle(rgb_cams[i], pts_px[i][idx], 3, rgb, -1)
            depth_cams[i] = cv2.circle(depth_cams[i], pts_px[i][idx], 3, (8000), -1)

    fig = plt.figure("Matched features")
    for i in range(len(rgb_cams)):
        ax = fig.add_subplot((len(rgb_cams)+2)//3, 3, i+1)
        imgplot = plt.imshow(rgb_cams[i])
        ax.set_title('Cam{}'.format(i))
    fig2 = plt.figure("Corresponding depth")
    for i in range(len(rgb_cams)):
        ax2 = fig2.add_subplot((len(rgb_cams)+2)//3, 3, i+1)
        imgplot2 = plt.imshow(depth_cams[i])
        ax2.set_title("Sensor{}".format(i))

    virtual_img = 255 * np.ones([320,240,3], dtype=np.uint8)
    # virtual_img = cv2.circle(virtual_img, q0v, 7, (255,0,0), -1)
    # virtual_img = cv2.circle(virtual_img, q1v, 7, (255,0,0), -1)
    # virtual_img = cv2.circle(virtual_img, q2v, 7, (255,0,0), -1)
    for idx in range(len(virtual_pts)):
        rgb = np.random.rand(3,)*255
        rgb_cams[virtual_view] = cv2.circle(rgb_cams[virtual_view], pts_px[virtual_view][idx], 4, rgb, -1)
        virtual_img = cv2.circle(virtual_img, (int(round(virtual_pts[idx][0])),int(round(virtual_pts[idx][1]))), 4, rgb, -1)
        rgb_cams[virtual_view] = cv2.circle(rgb_cams[virtual_view], (int(round(virtual_pts[idx][0])),int(round(virtual_pts[idx][1]))), 2, rgb-40, -1)

    fig = plt.figure("Result")
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(rgb_cams[virtual_view])
    ax.set_title('Ground truth (view n°{})'.format(virtual_view))
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(virtual_img)
    ax.set_title('Virtual image point placement')

    plt.show()

if __name__ == "__main__":
    main()