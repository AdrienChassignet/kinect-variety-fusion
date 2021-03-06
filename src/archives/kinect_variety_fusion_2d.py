from os import error
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from helpers import *
from fp_piv import *
import canny_edge_detector
import copy
from scipy.optimize import fsolve, minimize, Bounds, least_squares
from time import time
from scipy.spatial.transform import Rotation as R
from visualization import plot_F_squared_around_x
from functools import partial

VOXEL_SIZE = 0.025
VISUALIZE = False
VISUALIZE_FINAL = False

WINDOW_WIDTH = 320 #1280
WINDOW_HEIGHT = 240 #720
MAX_DEPTH = 5000

MATCH_RATIO = 0.71 #This influence how tolerant we will be to discriminate outlier matches, the lower the pickier 

FOLDER_NAME = "data/artificial_data/74/"
TIMESTAMP = "_20210629-1539"

#Intrinsics of the Kinect with the corresponding last 4 digits ID
CAM_INTRINSICS_4512 = o3d.camera.PinholeCameraIntrinsic(1280,720,612.6449585,612.76092529,635.7354126,363.57376099)

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

def image_points_selection(rgb_cams, depth_cams, by_hand=False):
    """
    Retrieve matching points among all the views and pick the 3 reference points.
    The 1st ref point Q0 is the centroid of the matched points, the 2nd point Q1 is the
    first extreme point in the lower hull and the 3rd point Q2 is the same with the upper hull.

    Inputs :    - rgb_cams: (n x rgb) List of the RGB images
                - depth_cams: (n x depth_map) List of the corresponding depth maps
    Outputs :   - q0: List of image position of Q0 in the input views
                - q1: List of image position of Q1 in the input views
                - q2: List of image position of Q2 in the input views
                - pts: List of image position of matched points in the input views
                - d_pts: List of depth corresponding to matched points in the input views
    """
    if by_hand:
        pts = [[(52,76), (134,179), (262,170), (148,10), (111,122), (262,94)],
                [(55,80), (132,184), (264,175), (151,14), (110,126), (263,97)],
                [(53,84), (126,189), (262,180), (151,17), (106,130), (259,101)],
                [(54,86), (126,195), (263,185), (151,19), (106,134), (260,106)],
                [(49,87), (127,207), (264,194), (148,18), (104,141), (265,115)]]
    else:
        pts = matched_points_extraction(rgb_cams, visualize=True)
        pts = common_points_extraction(pts)
    m = len(pts[0])

    d_pts, pts = depth_value_extraction(depth_cams, rgb_cams, pts)
    
    # arr_pts = []
    # common_pts = [[] for i in range(len(rgb_cams))]
    # for i in range(len(rgb_cams)):
    #     arr_pts.append(np.array(pts[i], dtype=object))
        # for idx in common_pts_idx:
        #     common_pts[i].append(pts[i][idx])

    # arr_pts = common_pts

    # Use the first view to define the reference points
    ref_view = 0 # len(rgb_cams)//2
    count = 0
    sum_u = 0
    sum_v = 0
    valid_ref_pts = []
    common_pts = [[] for i in range(len(rgb_cams))]
    common_d_pts = [[] for i in range(len(rgb_cams))]
    for idx, pt in enumerate(pts[ref_view]):
        if [] not in [pts[view][idx] for view in range(len(rgb_cams))]:
            count += 1
            sum_u += pt[0]
            sum_v += pt[1]
            valid_ref_pts.append(pt)
            for view_idx in range(len(rgb_cams)):
                common_pts[view_idx].append(pts[view_idx][idx])
                common_d_pts[view_idx].append(d_pts[view_idx][idx])

    pts = common_pts
    d_pts = common_d_pts

    centroid = (sum_u/count, sum_v/count)
    deltas = np.array(valid_ref_pts) - centroid
    centroid_idx = pts[ref_view].index(valid_ref_pts[np.argmin(np.einsum('ij,ij->i', deltas, deltas))])

    up_hull, low_hull = convex_hull(valid_ref_pts, split=True)
    for pt in up_hull:
        rgb_cams[0] = cv2.circle(rgb_cams[0], pt, 4, (0,255,0), -1)
    for pt in low_hull:
        rgb_cams[0] = cv2.circle(rgb_cams[0], pt, 4, (0,0,255), -1)
    # Select first extreme point in a counter-clockwise order in both lower and upper hull
    q1_idx = pts[ref_view].index(low_hull[-1])
    q2_idx = pts[ref_view].index(up_hull[-1])
    # centroid_idx = pts[ref_view].index(low_hull[0])
    # TODO: noncollinearity check!

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
        idxs = [centroid_idx, q1_idx, q2_idx]
        idxs.sort(reverse=True)
        for idx in idxs:
            rgb_cams[i] = cv2.circle(rgb_cams[i], pts[i][idx], 3, (255,0,0), -1)
            depth_cams[i] = cv2.circle(depth_cams[i], pts[i][idx], 2, (8000), -1)
            del pts[i][idx]
            del d_pts[i][idx]

    for idx in range(len(pts[0])):
        rgb = np.random.rand(3,)*255
        for i in range(len(rgb_cams)):
            rgb_cams[i] = cv2.circle(rgb_cams[i], pts[i][idx], 3, rgb, -1)
            depth_cams[i] = cv2.circle(depth_cams[i], pts[i][idx], 3, (8000), -1)

    print(len(pts[0]), pts[0])

    return q0, d0, q1, d1, q2, d2, pts, d_pts

def features_extraction(rgb_cams, max_feat=15000):
    detector = cv2.ORB_create(max_feat)
    descriptor = cv2.xfeatures2d.BEBLID_create(0.9)

    kp = [0 for i in range(len(rgb_cams))]
    des = [0 for i in range(len(rgb_cams))]
    for i, rgb_cam in enumerate(rgb_cams):
        kp[i] = (detector.detect(rgb_cam, None))
        kp[i], des[i] = descriptor.compute(rgb_cam, kp[i])
        des[i] = np.float32(des[i])

    return kp, des

def common_points_extraction(pts):
    """
    Return only the points that are common to each views
    """

    common_pts = [[] for i in range(len(pts))]
    for idx, pt in enumerate(pts[0]):
        if [] not in [pts[view][idx] for view in range(len(pts))]:
            for view_idx in range(len(pts)):
                common_pts[view_idx].append(pts[view_idx][idx])

    return common_pts

def features_matching_accross_all_pairs(kp, des, nn_match_ratio=MATCH_RATIO, max_feats=15000):
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matched_pts = [[[] for j in range(max_feats)] for i in range(len(kp))]
    max_idx = 0
    # Process all pairs of views
    for i in range(0, len(kp)):
        for j in range(i+1,len(kp)):
                matches = matcher.knnMatch(des[i],des[j],k=2)
                good_match_idx = 0
                for idx, (m, n) in enumerate(matches):
                    if m.distance < nn_match_ratio * n.distance:
                        # Good matches
                        pt1 = kp[i][m.queryIdx].pt
                        pt1 = (u1, v1) = (int(round(pt1[0])), int(round(pt1[1])))
                        pt2 = kp[j][m.trainIdx].pt
                        pt2 = (int(round(pt2[0])), int(round(pt2[1])))
                        try: # Check if the first feature point has already been selected
                            pt1_idx = matched_pts[i].index(pt1)
                            # If so, match the point in the 2nd view at the corresponding index
                            matched_pts[j][pt1_idx] = pt2 # WARNING: Maybe check if there is already a point here and if it is the same?
                            max_idx -= 1
                        except ValueError:
                            try: # Else check if the second feature point has already been selected
                                pt2_idx = matched_pts[j].index(pt2)
                                # If so, match the point in the 1st view at the corresponding index
                                matched_pts[i][pt2_idx] = pt1 # WARNING: Maybe check if there is already a point here and if it is the same?
                                max_idx -= 1
                            except ValueError:
                                matched_pts[i][max_idx + good_match_idx] = pt1
                                matched_pts[j][max_idx + good_match_idx] = pt2
                        good_match_idx += 1
                max_idx = max_idx + good_match_idx

    return matched_pts

def matched_points_extraction(rgb_cams, visualize=False):
    """
    Find the feature points that are common among the views.

    Inputs:     - rgb_cams: List of the RGB views.
    Outputs:    - pts: List for each view of a list of the matched points accross all the views. 
    """
    max_feat = 10000
    
    kp, des = features_extraction(rgb_cams, max_feat)

    test_pts = features_matching_accross_all_pairs(kp, des)

    return test_pts
        
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    pts = [[[] for j in range(max_feat)] for i in range(len(rgb_cams))]

    if visualize:
        res = [np.empty((WINDOW_HEIGHT, 2*WINDOW_WIDTH, 3), dtype=np.uint8) for i in range(len(rgb_cams)-1)]
    # The first camera view is taken as a reference
    for cam_idx in range(0, len(rgb_cams)-1):
        start_time_matching = time()
        matches = matcher.knnMatch(des[cam_idx],des[cam_idx+1],k=2)

        good_matches = []
        nn_match_ratio = 0.8  # Nearest neighbor matching ratio
        for i, (m, n) in enumerate(matches):
            if m.distance < nn_match_ratio * n.distance:
                good_matches.append(m)

        if visualize:
            print(len(good_matches), " good matches found.")
            cv2.drawMatches(rgb_cams[cam_idx], kp[cam_idx], rgb_cams[cam_idx+1], kp[cam_idx+1], good_matches, res[cam_idx])

        for idx in range(len(good_matches)):
            pt1 = kp[cam_idx][good_matches[idx].queryIdx].pt
            pt1 = (v1, u1) = (int(round(pt1[1])), int(round(pt1[0])))
            if cam_idx == 0: # The first pairs of match are kept
                pts[cam_idx][idx] = pt1
                pt2 = kp[cam_idx+1][good_matches[idx].trainIdx].pt
                pts[cam_idx+1][idx] = (int(round(pt2[1])), int(round(pt2[0])))
                # # Create the others pts list for the remaining views
                # for i in range(2, len(rgb_cams)):
                #     pts[i] = [[] for j in range(len(pts[0]))]
            else:
                try: 
                    t = pts[cam_idx].index(pt1)
                    # We populate the list of the currently processed view
                    pt2 = kp[cam_idx+1][good_matches[idx].trainIdx].pt
                    pts[cam_idx+1][t] = ((int(round(pt2[1])), int(round(pt2[0]))))
                except ValueError:
                    pass
        print("Feature matching between cam0 and cam", cam_idx, " in ", time()-start_time_matching, " seconds.")

    # Find the feature points that are not common to all the views
    common_pts = []
    # for i in range(2, len(rgb_cams)):
    #     if common_pts == []:
    #         common_pts = [idx for idx,x in enumerate(pts[i]) if x != []][::-1]
    #     else:
    #         for j, idx in enumerate(common_pts):
    #             if pts[i][idx] == []:
    #                 del common_pts[j]

    if visualize:
        fig = plt.figure("Features matching")
        for i in range(len(rgb_cams)-1):
            ax = fig.add_subplot(len(rgb_cams)//2, 2, i+1)
            imgplot = plt.imshow(res[i])
            ax.set_title('Cam{}/Cam{}'.format(i+1, i+2))
        plt.show()

    return pts, common_pts

def get_projective_depth(dmap, pts):
    """
    This method finds the projective depth of each given points.
    Projective is the distance of a scene point to the image plane on the 
    principal axis.

    Inputs :    - dmap: The depth image
                - pts: The list of points of reference
    Outputs :   - proj_d: The list of the projective depth of the points of reference given as input
    """

    d = np.zeros([WINDOW_HEIGHT,WINDOW_WIDTH], dtype=np.uint16)
    for (u,v) in pts:
        d[v,u] = dmap[v,u]
    pc = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(d), CAM_INTRINSICS_4512, depth_scale=1., depth_trunc=6000.)
    xyz = pcd2xyz(pc)
    z = xyz[-1]

    u_cop, v_cop = o3d.camera.PinholeCameraIntrinsic.get_principal_point(CAM_INTRINSICS_4512)
    d_cop = dmap[int(round(v_cop)), int(round(u_cop))] # TODO: Check for a valid depth value
    d = np.zeros([WINDOW_HEIGHT,WINDOW_WIDTH], dtype=np.uint16)
    d[int(round(v_cop)), int(round(u_cop))] = 100
    xyz_cop = pcd2xyz(o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(d), CAM_INTRINSICS_4512, depth_scale=1., depth_trunc=6000.))
    principal_axis = np.concatenate(xyz_cop)

    for (x,y,z) in xyz.T:
        pt_vec = np.array([x-u_cop])

    proj_d = np.zeros(len(pts))
    sorted_pts = sorted(pts , key=lambda k: [k[1], k[0]])
    for i, pt in enumerate(pts):
        proj_d[i] = z[sorted_pts.index(pt)]


    return proj_d

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
        depth = np.zeros(len(pts_list))
        valid = True
        for i in range(len(pts_list)): # Check depth of current point in each view
            if pts_list[i][idx] != []:
                (u,v) = pts_list[i][idx]
                neighborhood = get_neighborhood(u, v, 0, dmap_list[i])
                nonzero = neighborhood[np.nonzero(neighborhood)]
                count = len(nonzero)
                if count > 0: # and (max(nonzero) - min(nonzero)) < 100:
                    depth[i] = sorted(nonzero)[count//2] #Take median value
                else:
                    valid = False
                    break
        if valid: # If there is valid depth information in all views we keep the point
            for i in range(len(pts_list)):
                pts_depth[i].append(depth[i])
                updated_pts[i].append(pts_list[i][idx])

    # proj_d = get_projective_depth(dmap_list[0], updated_pts[0])

    return pts_depth, updated_pts

def normalize_uvd(q0, d0, q1, d1, q2, d2, pts, d_pts):
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
        q0_n.append((q0[i][0]/WINDOW_WIDTH, q0[i][1]/WINDOW_WIDTH))
        d0_n.append(d0[i]/MAX_DEPTH)
        q1_n.append((q1[i][0]/WINDOW_WIDTH, q1[i][1]/WINDOW_WIDTH))
        d1_n.append(d1[i]/MAX_DEPTH)
        q2_n.append((q2[i][0]/WINDOW_WIDTH, q2[i][1]/WINDOW_WIDTH))
        d2_n.append(d2[i]/MAX_DEPTH)
        for j, pt in enumerate(pts[i]):
            pts_n[i].append((pt[0]/WINDOW_WIDTH, pt[1]/WINDOW_WIDTH))
            d_pts_n[i].append(d_pts[i][j]/MAX_DEPTH)

    return q0_n, d0_n, q1_n, d1_n, q2_n, d2_n, pts_n, d_pts_n

#----------------------------------------------------------------------------------------------

def main():
    start_time = time()

    rgb_cams = []
    depth_cams = []
    cams_idx = range(600,800,25)
    for idx in cams_idx:
        # if i != 2: # keep view 2 for reconstruction
        rgb_cam, depth_cam = load_rgbd2(str(idx))
        rgb_cams.append(rgb_cam)
        depth_cams.append(depth_cam)
        #TODO: get the maximum depth value to set upper bound on solution space
    print("Loaded data in ", time()-start_time, " seconds.")

    start_time = time()
    q0, d0, q1, d1, q2, d2, pts, d_pts = image_points_selection(rgb_cams, depth_cams, by_hand=False)
    pts_px = pts.copy()
    # Normalize the pixel position and depth to be on a [0,1] scale for a robust optimization
    q0, d0, q1, d1, q2, d2, pts, d_pts = normalize_uvd(q0, d0, q1, d1, q2, d2, pts, d_pts)
    print("Points selection in ", time()-start_time, " seconds.")
    print(len(pts[0]))

    virtual_cam = 675
    virtual_view = cams_idx.index(virtual_cam)
    print("Reconstructing camera view number ", virtual_cam)
    q0v = q0[virtual_view]
    q1v = q1[virtual_view]
    q2v = q2[virtual_view]

    # coeffs = []
    # resids = []
    time_start = time()
    # for pt_idx in range(len(pts[0])):
    #     Kis = []
    #     for view in range(len(pts)):
    #         if view != virtual_view and pts[view][pt_idx] != []:
    #             Kis.append(build_constraints_matrix(q0[view], q1[view], q2[view], d0[view], d1[view], d2[view], pts[view][pt_idx], d_pts[view][pt_idx]))
    #     Kis = np.concatenate(Kis) 
    #     _, D, V = np.linalg.svd(Kis)
    #     coeffs.append(V[-1]/V[-1,-1])
    #     print("Residual error: ", np.matmul(Kis, coeffs[-1]))
    #     print("Residual error sum of squares: ", np.sum(np.matmul(Kis, coeffs[-1])**2))
    #     resids.append(np.sum(np.matmul(Kis, coeffs[-1])**2))
    #     # print("Time: ", time()-time_start, "Structure coefficients: ", coeffs[-1])

    coeffs, resids = get_structure_coefficients(q0, q1, q2, d0, d1, d2, pts, d_pts)
    print("Time computing parameters: ", time()-time_start)

    vertices = [] # format [u,v,d]
    error_img_pos = np.zeros(0)
    error_depth = np.zeros(0)
    error_u = np.zeros(0)
    error_v = np.zeros(0)
    results = []
    csts = []
    virtual_pts = []
    virtual_img = 255 * np.ones([WINDOW_HEIGHT,WINDOW_WIDTH,3], dtype=np.uint8)
    for pt_idx in range(len(pts[0])):
        # Estimate the position of the current processed point in the new virtual view
        qv = pts[virtual_view][pt_idx]
        if qv != []:
            (uv, vv) = qv
            # expected = np.array([uv, vv, d_pts[virtual_view][i]/d0[virtual_view], d1[virtual_view]/d0[virtual_view], d2[virtual_view]/d0[virtual_view]])
            expected = np.array([uv, vv, d_pts[virtual_view][pt_idx], 1, 1])
            
            # Start the search from a close position in a reference view
            ref_view = virtual_view+1
            while ref_view == virtual_view or pts[ref_view][pt_idx] == []:
                ref_view += 1
            # x0 = np.array([pts[ref_view][i][0], pts[ref_view][i][1], d1[ref_view]/d0[ref_view], d2[ref_view]/d0[ref_view], d_pts[ref_view][i]/d0[ref_view]])
            x0 = np.array([pts[ref_view][pt_idx][0], pts[ref_view][pt_idx][1], d_pts[ref_view][pt_idx], 1, 1])
            # x0 = np.array([WINDOW_WIDTH//2, WINDOW_HEIGHT//2, .1, .1, .1])
            # x0 = expected
            # print("Initialization [u, v, g1, g2, g3] = ", x0)

            cst = [q0v[0],q0v[1],q1v[0],q1v[1],q2v[0],q2v[1],coeffs[pt_idx], d0[virtual_view], d1[virtual_view], d2[virtual_view]]
            csts.append(cst)
            # x = fsolve(F, x0, args=cst)
            # error_norm = np.linalg.norm(expected - x, ord=2)
            # assert error_norm < tol, 'norm of error =%g' % error_norm
            # print("F(x) = ", F(x, cst))

            # Minimize sum of squares to solve the system

            # Define bounds of the solution space TODO: use max depth value of inputs with security margin
            b = [[0,0,0,-np.inf,-np.inf], [WINDOW_WIDTH,WINDOW_HEIGHT,6000,np.inf,np.inf]]
            b = [[0,0,0,-np.inf,-np.inf], [1,1,2,np.inf,np.inf]]
            # res = {'x': [0,0,0,0,0], 'fun': 0}
            # res['x'], res['fun'] = minimize_brute_force(x0, cst, b, px_width=30)
            # b = [[x0[0]-100, x0[1]-100, x0[2]-500, 1, 1], [x0[0]+100, x0[1]+100, x0[2]+500, 1, 1]]

            # partial_F_3var = partial(call_F_3var_arg, F_3var, cst)
            # res = least_squares(partial_F_3var, x0, bounds=b, method='trf', loss='linear')

            # Convert bounds into constraints
            cons = []
            for idx in range(len(b[0])):
                lower = b[0][idx]
                upper = b[1][idx]
                l = {'type':'ineq', 'fun': lambda x, lb=lower, i=idx: x[i] - lb}
                u = {'type':'ineq', 'fun': lambda x, ub=upper, i=idx: ub - x[i]}
                cons.append(l)
                cons.append(u)
            b = Bounds(b[0], b[1], False)

            res = minimize(sum_of_squares_of_F_3var_norm_fact, x0, cst, method='Nelder-mead', constraints=cons, bounds=b, options={'disp': True})
            # print(pt_idx, "- ", res["success"])
            if res:
                u = res["x"][0] 
                v = res["x"][1]
                d = res["x"][2]
                error_img_pos = np.append(error_img_pos, np.linalg.norm(np.array([uv,vv])*WINDOW_WIDTH - np.array([u,v])*WINDOW_WIDTH, ord=2))
                error_depth = np.append(error_depth, abs(d_pts[virtual_view][pt_idx]*MAX_DEPTH - d*MAX_DEPTH))
                error_u = np.append(error_u, (uv-u)*WINDOW_WIDTH)
                error_v = np.append(error_v, (vv-v)*WINDOW_WIDTH)
                results.append(res['x'])
                virtual_pts.append((u*WINDOW_WIDTH,v*WINDOW_WIDTH))

                print("Expected result: ", expected)
                print("Solution found: ", res["x"], " / residual being: ", res["fun"])
                print("Error px: ", error_img_pos[-1])
                print('--------------------------------------------------')

            # if res["fun"] < .001: # or error_norm < 50:
            #     print("\nExpected u and v in the new view: ", expected)
            #     # print("Solution with fsolve: ", x)
            #     # print('norm of error =%g' % error_norm)
            #     print("Least Squares => \nx=", res["x"], "\nResidual:", res["fun"])
            #     print("Resiudal of parameters estimation was: ", resids[i])
            #     print('--------------------------------------------------')

    for idx in range(len(error_img_pos)):
        print("{4}- Point {0} ; Residual error for coeffs: {1} ; Pixel error: {2} ; Depth error: {3}".format(pts_px[virtual_view][idx], resids[idx], error_img_pos[idx], error_depth[idx], idx))

    print("Time recomputing {} point positions: ".format(len(error_img_pos)), time()-time_start)
    print("Error in pixels for point placement: \nMean: {0} / Max: {1} / Min: {2} / Std: {3}".format(np.mean(error_img_pos),
            np.max(error_img_pos), np.min(error_img_pos), np.std(error_img_pos)))
    print("Error in mm for corresponding depth: \nMean: {0} / Max: {1} / Min: {2} / Std: {3}".format(np.mean(error_depth),
            np.max(error_depth), np.min(error_depth), np.std(error_depth)))


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
    ax.set_title('Ground truth (view n??{})'.format(virtual_view))
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(virtual_img)
    ax.set_title('Virtual image point placement')

    # for i, res in enumerate(results):
    #     if i in [0,1,2]:
    #         fig = plot_F_squared_around_x(res, csts[i], name="Plots of point {}".format(i))
        # if error_img_pos[-1] > 1:
        #     fig2 = plot_F_squared_around_x(expected, cst, name="F?? around expected solution")

    plt.show()

    print("Finished")


if __name__ == "__main__":
    main()