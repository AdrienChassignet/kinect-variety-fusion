from math import nan
import cv2
import numpy as np
from matplotlib import pyplot as plt
from helpers import *
from fp_piv import *
import canny_edge_detector
import copy
from scipy.optimize import fsolve, minimize
from time import time
from scipy.spatial.transform import Rotation as R

VOXEL_SIZE = 0.025
VISUALIZE = False
VISUALIZE_FINAL = False

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

FOLDER_NAME = "data/multicap_smallB_bed/"
TIMESTAMP = "_20210623-1903"

def load_rgbd(idx):
    rgb_cam = cv2.imread(FOLDER_NAME+"rgb_"+idx+TIMESTAMP+".jpg", cv2.IMREAD_COLOR)
    rgb_cam = cv2.cvtColor(rgb_cam, cv2.COLOR_BGR2RGB)
    depth_cam = np.load(FOLDER_NAME+"depth_"+idx+TIMESTAMP+".npy")

    return rgb_cam, depth_cam

def load_images(visu=False):
    rgb_cam1 = cv2.imread(FOLDER_NAME+"rgb_1"+TIMESTAMP+".jpg", cv2.IMREAD_COLOR)
    rgb_cam1 = cv2.cvtColor(rgb_cam1, cv2.COLOR_BGR2RGB)
    depth_cam1 = np.load(FOLDER_NAME+"depth_1"+TIMESTAMP+".npy")
    rgb_cam2 = cv2.imread(FOLDER_NAME+"rgb_2"+TIMESTAMP+".jpg", cv2.IMREAD_COLOR)
    rgb_cam2 = cv2.cvtColor(rgb_cam2, cv2.COLOR_BGR2RGB)
    depth_cam2 = np.load(FOLDER_NAME+"depth_2"+TIMESTAMP+".npy")
    
    if visu:
        fig = plt.figure("Input views")
        ax = fig.add_subplot(2, 2, 1)
        imgplot = plt.imshow(rgb_cam1)
        ax.set_title('RGB Cam1')
        ax = fig.add_subplot(2, 2, 2)
        imgplot = plt.imshow(rgb_cam2)
        ax.set_title('RGB Cam2')
        ax = fig.add_subplot(2, 2, 3)
        imgplot = plt.imshow(depth_cam1)
        ax.set_title('Depth Cam1')
        ax = fig.add_subplot(2, 2, 4)
        imgplot = plt.imshow(depth_cam2)
        ax.set_title('Depth Cam2')
        plt.show()

    return rgb_cam1, depth_cam1, rgb_cam2, depth_cam2

def image_points_selection(rgb_cams, depth_cams):
    """
    Retrieve matching points among all the views and pick the 3 reference points.
    The 1st ref point Q0 is the centroid of the matched points, the 2nd point Q1 is the
    first extreme point in the lower hull and the 3rd point Q2 is the same with the upper hull.

    Inputs :    - rgb_cams: (n x rgb) List of the RGB images
                - depth_cams: (n x depth_map) List of the corresponding depth maps
    Outputs :   - q0: List of image position of Q0 in the input views
                - q1: List of image position of Q1 in the input views
                - q2: List of image position of Q2 in the input views
                - arr_pts: List of image position of matched points in the input views
                - d_pts: List of depth corresponding to matched points in the input views
    """

    pts = feature_matching_2d(rgb_cams, visualize=True)
    m = len(pts[0])

    d_pts, pts = depth_value_extraction(depth_cams, rgb_cams, pts)
    
    arr_pts = []
    for i in range(len(rgb_cams)):
        arr_pts.append(np.array(pts[i]))

    # Use the first view to define the reference points
    centroid = (np.sum(arr_pts[0][:,0])/m, np.sum(arr_pts[0][:,1])/m)
    deltas = arr_pts[0] - centroid
    centroid_idx = np.argmin(np.einsum('ij,ij->i', deltas, deltas))
    low_hull1, up_hull1 = convex_hull(pts[0], split=True)
    q1_idx = np.where(arr_pts[0] == low_hull1[0])[0][0]
    q2_idx = np.where(arr_pts[0] == up_hull1[0])[0][0]

    q0 = []
    d0 = []
    q1 = []
    d1 = []
    q2 = []
    d2 = []
    for i in range(len(rgb_cams)):
        q0.append(pts[i][centroid_idx])
        d0.append(d_pts[i][centroid_idx])
        q1.append(pts[i][q1_idx])
        d1.append(d_pts[i][q1_idx])
        q2.append(pts[i][q2_idx])
        d2.append(d_pts[i][q2_idx])
        arr_pts[i] = np.delete(arr_pts[i] , [centroid_idx, q1_idx, q2_idx], axis=0)
        d_pts[i] = np.delete(d_pts[i] , [centroid_idx, q1_idx, q2_idx], axis=0)

    return q0, d0, q1, d1, q2, d2, arr_pts, d_pts

def feature_matching_2d(rgb_cams, visualize=False):
    """
    Find the feature points that are common among the views.
    This implementation seeks for feature points that appears in ALL the views.

    Inputs:     - rgb_cams: List of the RGB views. The first view is taken as a reference.
    Outputs:    - pts: List for each view of a list of the matched points accross all the views. 
    """

    detector = cv2.ORB_create(30000)
    descriptor = cv2.xfeatures2d.BEBLID_create(0.9)

    kp = [0 for i in range(len(rgb_cams))]
    des = [0 for i in range(len(rgb_cams))]
    for i, rgb_cam in enumerate(rgb_cams):
        kp[i] = (detector.detect(rgb_cam, None))
        kp[i], des[i] = descriptor.compute(rgb_cam, kp[i])
        des[i] = np.float32(des[i])
        
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    pts = [[] for i in range(len(rgb_cams))]

    if visualize:
        res = [np.empty((WINDOW_HEIGHT, 2*WINDOW_WIDTH, 3), dtype=np.uint8) for i in range(len(rgb_cams)-1)]
    # The first camera view is taken as a reference
    for cam_idx in range(1, len(rgb_cams)):
        start_time_matching = time()
        matches = matcher.knnMatch(des[0],des[cam_idx],k=2)

        good_matches = []
        nn_match_ratio = 0.8  # Nearest neighbor matching ratio
        for i, (m, n) in enumerate(matches):
            if m.distance < nn_match_ratio * n.distance:
                good_matches.append(m)

        if visualize:
            print(len(good_matches), " good matches found.")
            cv2.drawMatches(rgb_cams[0], kp[0], rgb_cams[cam_idx], kp[cam_idx], good_matches, res[cam_idx-1])

        # Filter the matched points to keep only the common accross all views
        for idx in range(len(good_matches)):
            pt1 = kp[0][good_matches[idx].queryIdx].pt
            pt1 = (v1, u1) = (int(round(pt1[1])), int(round(pt1[0])))
            if cam_idx == 1: # The first pairs of match are kept
                pts[0].append(pt1)
                pt2 = kp[cam_idx][good_matches[idx].trainIdx].pt
                pts[cam_idx].append((int(round(pt2[1])), int(round(pt2[0]))))
                # Create the others pts list for the remaining views
                for i in range(2, len(rgb_cams)):
                    pts[i] = [[] for j in range(len(pts[0]))]
            else:
                # neighborhood = [(v1-1, u1-1),(v1-1, u1), (v1-1, u1+1),
                #                 (v1, u1-1), pt1, (v1, u1+1),
                #                 (v1+1, u1-1), (v1+1, u1), (v1+1, u1+1)]
                # while pts[0][idx] not in neighborhood: # If the point is not common remove it in all other lists
                #     for j in range(0, cam_idx):
                #         del pts[j][idx]
                #     if idx >= len(pts[0]):
                #         break
                # if idx >= len(pts[0]):
                #     break
                try: 
                    t = pts[0].index(pt1)
                    # We populate the list of the currently processed view
                    pt2 = kp[cam_idx][good_matches[idx].trainIdx].pt
                    pts[cam_idx][t] = ((int(round(pt2[1])), int(round(pt2[0]))))
                except ValueError:
                    pass
        print("Feature matching between cam0 and cam", cam_idx, " in ", time()-start_time_matching, " seconds.")

    for i in range(2, len(rgb_cams)):
        no_match_idx = [idx for idx,x in enumerate(pts[i]) if x == []][::-1]
        for idx in no_match_idx:
            for j in range(len(rgb_cams)):
                del pts[j][idx]

    if visualize:
        fig = plt.figure("Features matching")
        for i in range(len(rgb_cams)-1):
            ax = fig.add_subplot(len(rgb_cams)//2, 2, i+1)
            imgplot = plt.imshow(res[i])
            ax.set_title('Cam1/Cam{}'.format(i+2))
        plt.show()

    return pts

def depth_value_extraction(dmap_list, image_list, pts_list):
    """
        Extract valid depth of each matched points accross the views.
        The inputs and outputs are list of corresponding data of each view.
        The method extracts the mean of a 3x3 neighborhood in the depth map for each matched
        points and discard the point if no valid depth is available.

        Inputs :    - dmap_list: (n x dmap)List of the depth maps
                    - image_list: (n x rgb_image) List of the RGB images for potential edge detection
                    - pts_list: (n x nb_matched_points) List of matched points in each views
        Outputs :   - pts_depth : (n x nb_valid_points) List of depths of the matched valid points
                    - updated_pts : (n x nb_valid_points) List of matched valid points
    """
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5,5), 0)
    # v = np.median(image)
    # # apply automatic Canny edge detection using the computed median
    # lower = int(max(0, (1.0 - 0.6) * v))
    # upper = int(min(255, (1.0 + 0.6) * v))
    # edged = cv2.Canny(blurred, lower, upper)
    # plt.figure(figsize=(15, 5))
    # plt.imshow(edged)
    # plt.show()

    updated_pts = [[] for i in range(len(pts_list))]
    pts_depth = [[] for i in range(len(pts_list))]

    for idx in range(len(pts_list[0])): # Check all matched points
        means = np.zeros(len(pts_list))
        for i in range(len(pts_list)): # Check depth of current point in each view
            (u,v) = pts_list[i][idx]
            neighborhood = np.array([dmap_list[i][u-1][v-1], dmap_list[i][u-1][v], dmap_list[i][u-1][v+1],
                                dmap_list[i][u][v-1], dmap_list[i][u][v], dmap_list[i][u][v+1],
                                dmap_list[i][u+1][v-1], dmap_list[i][u+1][v], dmap_list[i][u+1][v+1]])
            means[i] = neighborhood[np.nonzero(neighborhood)].mean()
        if np.sum(means) > 0: # If there is valid depth information in all views we keep the point
            for i in range(len(pts_list)):
                pts_depth[i].append(means[i])
                updated_pts[i].append(pts_list[i][idx])

    return pts_depth, updated_pts


#----------------------------------------------------------------------------------------------

def main():
    # rgb_cam1, depth_cam1, rgb_cam2, depth_cam2 = load_images(visu=False)
    start_time = time()

    rgb_cams = []
    depth_cams = []
    N = 7
    for i in range(0, N):
        rgb_cam, depth_cam = load_rgbd(str(i))
        rgb_cams.append(rgb_cam)
        depth_cams.append(depth_cam)
    print("Loaded data in ", time()-start_time, " seconds.")

    start_time = time()
    q0, d0, q1, d1, q2, d2, arr_pts, d_pts = image_points_selection(rgb_cams, depth_cams)
    print("Points selection in ", time()-start_time, " seconds.")
    print(len(arr_pts[0]))

    q0v = q0[1]
    q1v = q1[1]
    q2v = q2[1]

    coeffs = []
    correct_points = []
    time_start = time()
    for i in range(len(arr_pts[0])):
        # time_start = time()
        # print('--------------------------------------------------')
        # print("Processing the point ", i)
        # print('--------------------------------------------------')

        Kis = []
        for j in range(len(arr_pts)):
            Kis.append(build_constraints_matrix(q0[j], q1[j], q2[j], d0[j], d1[j], d2[j], arr_pts[j][i], d_pts[j][i]))
        Kis = np.concatenate(Kis) 
        _, D, V = np.linalg.svd(Kis)
        if D[-1] < 10:
            print("Residuals D: ", D)
            correct_points.append(i)
        coeffs.append(V[-1]/V[-1,-1])
        # print("Time: ", time()-time_start, "Structure coefficients: ", coeffs[-1])
    print("Time computing parameters: ", time()-time_start)
    print(correct_points)

    time_start = time()
    for i in correct_points:
        # Estimate the position of the current processed point in the new virtual view
        qv = (vv, uv) = arr_pts[1][i]
        expected = np.array([uv, vv, d1[1]/d0[1], d2[1]/d0[1], d_pts[1][i]/d0[1]])
        print("\nExpected u and v in the new view: ", expected)
        
        # Start the search from the position in reference view
        x0 = np.array([arr_pts[0][i][1], arr_pts[0][i][0], d1[0]/d0[0], d2[0]/d0[0], d_pts[0][i]/d0[0]])
        # x0 = np.array([50, 50, .1, .1, .1])
        # print("Initialization [u, v, g1, g2, g3] = ", x0)

        cst = [q0v[1],q0v[0],q1v[1],q1v[0],q2v[1],q2v[0],coeffs[i]]
        x = fsolve(F, x0, args=cst)
        print("Time: ", time()-time_start, "Solution with fsolve: ", x)
        error_norm = np.linalg.norm(expected - x, ord=2)
        # assert error_norm < tol, 'norm of error =%g' % error_norm
        print('norm of error =%g' % error_norm)
        # print("F(x) = ", F(x, cst))

        time_start = time()
        # Test LS to solve the system
        res = minimize(sum_of_squares_of_F, x0, cst)
        print("LS => time: ", time()-time_start, "\nx=", res["x"], res["fun"])
        print('--------------------------------------------------')

    print("Time recomputing point positions: ", time()-time_start)


if __name__ == "__main__":
    main()