from tokenize import Comment
import open3d as o3d
import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
from helpers import *
import copy
from scipy.optimize import fsolve, minimize
from scipy.ndimage.filters import gaussian_filter
from time import time
from bisect import bisect
from scipy.spatial.transform import Rotation as R

VOXEL_SIZE = 0.025
VISUALIZE = False
VISUALIZE_FINAL = False

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

FOLDER_NAME = "messybed_wide_baseline/"
TIMESTAMP = "_20210618-1458"

#Intrinsics of the Kinect with the corresponding last 4 digits ID
CAM_INTRINSICS_4512 = o3d.camera.PinholeCameraIntrinsic(1280,720,612.6449585,612.76092529,635.7354126,363.57376099)
K_4512 = CAM_INTRINSICS_4512.intrinsic_matrix
SUB_CAM_INTRINSICS = o3d.camera.PinholeCameraIntrinsic(1280,720,611.85577393,611.68011475,638.63598633,369.67144775)
K_SUB = SUB_CAM_INTRINSICS.intrinsic_matrix

def load_pointclouds(rgb, depth, cam_intrinsics):
    # Load rgbd images into point clouds

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb), o3d.geometry.Image(depth), depth_trunc = 4., convert_rgb_to_intensity = False)
    pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_intrinsics)

    depth[depth == 0] = np.max(depth)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb), o3d.geometry.Image(depth), depth_trunc = 20.)
    pcd_full = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam_intrinsics)

    # Must compare here if doing the computation directly with the "full" pcd is faster than creating 2 duplicate pointclouds

    return pcd_raw, pcd_full

def test_correspondance(pcd):
    def get_gradient_2d(start, stop, width, height, is_horizontal):
        if is_horizontal:
            return np.tile(np.linspace(start, stop, width), (height, 1))
        else:
            return np.tile(np.linspace(start, stop, height), (width, 1)).T

    def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
        result = np.zeros((height, width, len(start_list)), dtype=np.float)

        for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
            result[:, :, i] = get_gradient_2d(start, stop, width, height, is_horizontal)

        return result

    colors = get_gradient_3d(WINDOW_WIDTH, WINDOW_HEIGHT, (0,0,192), (255,255,64), (True, False, False))
    colors = np.uint8(colors)
    plt.imshow(colors)
    plt.show()
    print("Nb points: ", len(pcd.colors), " // Nb pixels: ", len(colors))
    print(colors[500][0]/255)
    for i in range(len(pcd.colors)):
        pcd.colors[i] = colors[i//WINDOW_WIDTH][i%WINDOW_WIDTH]/255

    o3d.visualization.draw_geometries([pcd], width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    return

def downsample_pointclouds(pc_raw):
    # voxel downsample input pointcloud

    pcd = pc_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    
    print("Nb points before removal: ", np.asarray(pcd.points).size)
    # Remove outliers that are far from other points
    pcd, _ = pcd.remove_statistical_outlier(100, .2)
    print("Nb points after removal: ", np.asarray(pcd.points).size)

    return pcd

def features_matching(pc1, pc2):
    A_xyz = pcd2xyz(pc1) # np array of size 3 by N
    B_xyz = pcd2xyz(pc2) # np array of size 3 by M

    # extract FPFH features
    A_feats = extract_fpfh(pc1,VOXEL_SIZE)
    B_feats = extract_fpfh(pc2,VOXEL_SIZE)

    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(
        A_feats, B_feats, mutual_filter=True)
    A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
    B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs

    num_corrs = A_corr.shape[1]
    # print(f'FPFH generates {num_corrs} putative correspondences.')

    return A_corr, B_corr, num_corrs

def draw_features_correspondences(pc1, pc2, pc1_corr, pc2_corr, num_corrs):
    # visualize the point clouds together with feature correspondences

    points = np.concatenate((pc1_corr.T,pc2_corr.T),axis=0)
    lines = []
    for i in range(num_corrs):
        lines.append([i,i+num_corrs])
    colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pc1,pc2,line_set], width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

def global_registration(pc1_corr, pc2_corr):
    # robust global registration using TEASER++

    NOISE_BOUND = VOXEL_SIZE
    teaser_solver = get_teaser_solver(NOISE_BOUND)
    teaser_solver.solve(pc1_corr,pc2_corr)
    solution = teaser_solver.getSolution()
    # # Print the solution
    # print("Solution is:", solution)
    R_teaser = solution.rotation
    t_teaser = solution.translation

    return Rt2T(R_teaser,t_teaser)

def local_refinement(pc1, pc2, T_initial):
    # local refinement using ICP

    NOISE_BOUND = VOXEL_SIZE
    icp_sol = o3d.pipelines.registration.registration_icp(
        pc1, pc2, NOISE_BOUND, T_initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

    return icp_sol.transformation

def get_correspondences_pixel_coordinates(pc_corr, pc_full):
    """
    This function returns the array of the pixel coordinate of given reference
    points in pointcloud.
    Input:  - pc_corr: The point cloud of the correspondences in a list of xyz coordinates
            - pc_full: The point cloud of ALL the image points (its number of
                    points must match the number of pixels of the input image)
    Return: - px_coords: The list of the (v,u) coordinates of the corr points
    """
        
    pc_full = pcd2xyz(pc_full)

    px_idx, _ = find_correspondences(np.transpose(pc_full), pc_corr, mutual_filter=True)
    px_coords = list(zip(px_idx//WINDOW_WIDTH, px_idx%WINDOW_WIDTH))

    # for point in np.transpose(pc_corr):
    #     deltas = pc_full - point
    #     idx = np.argmin(np.einsum('ij,ij->i', deltas, deltas))
    #     # idx = np.where((pc_full[:,0] == point[0]) & (pc_full[:,1] == point[1]) & (pc_full[:,2] == point[2]))[0]
    #     if idx:
    #         px_coords.append((int(idx)//WINDOW_WIDTH, int(idx)%WINDOW_WIDTH))

    return px_coords

def remove_correspondences_outliers(A_corr, B_corr, T, dist_threshold=.01, max_dist=5.):
    """
    Remove correspondences outliers based on euclidean distance in point clouds and z coordinate
    Inputs: - A_corr xyz coordinates of the correspondance in the first view
            - B_corr xyz coordinates of the correspondance in the second view
            - T extrinsic transformation between the two views
            - dist_threshold
    Outputs:    - A_inliers:
                - B_inliers:
    """
    # if np.shape(A_corr)[0] != 3:
    #     A_corr = np.transpose(A_corr)
    # if np.shape(B_corr)[0] != 3:
    #     B_corr = np.transpose(B_corr)
    A_corr_T = pcd2xyz(copy.deepcopy(xyz2pcd(np.transpose(A_corr))).transform(T))
    A_inliers = []
    B_inliers = []
    for i in range(np.shape(A_corr)[1]):
        if (0 < A_corr_T[2,i] < max_dist) & (0 < B_corr[2,i] < max_dist):
            if np.sqrt(np.sum((A_corr_T[:2,i] - B_corr[:2,i])**2, axis=0)) < dist_threshold:
                A_inliers.append(A_corr[:,i])
                B_inliers.append(B_corr[:,i])

    return A_inliers, B_inliers

def get_common_correspondences(A_corr, B_corr, dist_threshold=.1):
    common_points = []

    for i in range(np.shape(A_corr)[0]): 
        for j in range(np.shape(B_corr)[0]): 
            if np.sqrt(np.sum((A_corr[i] - B_corr[j])**2, axis=0)) < dist_threshold:
                common_points.append(A_corr[i])

    return common_points

def build_constraints_matrix(q0, q1, q2, d0, d1, d2, q, d):
    # q = (v, u)
    g1 = d1/d0
    g2 = d2/d0
    g3 = d/d0
    a = g1*q1[1] - q0[1]
    b = g2*q2[1] - q0[1]
    c = g3*q[1] - q0[1]
    l = g1*q1[0] - q0[0]
    m = g2*q2[0] - q0[0]
    n = g3*q[0] - q0[0]
    r = g1 - 1
    s = g2 - 1
    t = g3 - 1

    return np.array([
        [a**2 - l**2, 2*a*b - 2*l*m, b**2 - m**2, 2*a*c - 2*l*n, 2*b*c - 2*m*n, c**2 - n**2],
        [l**2 - r**2, 2*l*m - 2*r*s, m**2 - s**2, 2*l*n - 2*r*t, 2*m*n - 2*s*t, n**2 - t**2],
        [a*l, b*l + a*m, m*b, c*l + a*n, c*m + b*n, n*c],
        [a*r, b*r + a*s, s*b, c*r + a*t, c*s + b*t, t*c],
        [l*r, m*r + l*s, s*m, n*r + l*t, n*s + m*t, t*n]
    ])

def F(x, cst):
    """
    Define the system of equations.
    cst = [u0, v0, u1, v1, u2, v2, coeffs]
    x = [u, v, g1, g2, g3]
    """
    [u0, v0, u1, v1, u2, v2, coeffs] = cst
    [u, v, g1, g2, g3] = x
    a = g1*u1 - u0
    b = g2*u2 - u0
    c = g3*u - u0
    l = g1*v1 - v0 
    m = g2*v2 - v0
    n = g3*v - v0
    r = g1 - 1
    s = g2 - 1
    t = g3 - 1
    return np.array([
        coeffs[0]*(a**2-l**2) + 2*coeffs[1]*(a*b-l*m) + coeffs[2]*(b**2-m**2) + 2*coeffs[3]*(a*c-l*n) + 2*coeffs[4]*(b*c-m*n) + c**2 - n**2,
        coeffs[0]*(l**2-r**2) + 2*coeffs[1]*(l*m-r*s) + coeffs[2]*(m**2-s**2) + 2*coeffs[3]*(l*n-r*t) + 2*coeffs[4]*(m*n-s*t) + n**2 - t**2,
        coeffs[0]*a*l + coeffs[1]*(l*b+m*a) + coeffs[2]*m*b + coeffs[3]*(l*c+n*a) + coeffs[4]*(m*c+b*n) + c*n,
        coeffs[0]*a*r + coeffs[1]*(r*b+s*a) + coeffs[2]*s*b + coeffs[3]*(r*c+t*a) + coeffs[4]*(s*c+b*t) + c*t,
        coeffs[0]*r*l + coeffs[1]*(l*s+m*r) + coeffs[2]*m*s + coeffs[3]*(l*t+n*r) + coeffs[4]*(m*t+s*n) + t*n   
    ])

def sum_of_squares_of_F(x, cst):
    return np.sum(F(x,cst)**2)

#----------------------------------------------------------------------------------------------

def main():

    start_time = time()
    sigma_x = sigma_y = 0
    sigma = [sigma_x, sigma_y]

    rgb_cam1 = cv2.imread("data/"+FOLDER_NAME+"rgb_1"+TIMESTAMP+".jpg", cv2.IMREAD_COLOR)
    rgb_cam1 = cv2.cvtColor(rgb_cam1, cv2.COLOR_BGR2RGB)
    depth_cam1 = np.load("data/"+FOLDER_NAME+"depth_1"+TIMESTAMP+".npy")
    depth_cam1 = gaussian_filter(depth_cam1, sigma, mode='constant')

    A_pcd_raw, A_pcd_full = load_pointclouds(rgb_cam1, depth_cam1, cam_intrinsics=CAM_INTRINSICS_4512)
    
    rgb_cam2 = cv2.imread("data/"+FOLDER_NAME+"rgb_2"+TIMESTAMP+".jpg", cv2.IMREAD_COLOR)
    rgb_cam2 = cv2.cvtColor(rgb_cam2, cv2.COLOR_BGR2RGB)
    depth_cam2 = np.load("data/"+FOLDER_NAME+"depth_2"+TIMESTAMP+".npy")
    depth_cam2 = gaussian_filter(depth_cam2, sigma, mode='constant')

    B_pcd_raw, B_pcd_full = load_pointclouds(rgb_cam2, depth_cam2, cam_intrinsics=CAM_INTRINSICS_4512)

    rgb_cam3 = cv2.imread("data/"+FOLDER_NAME+"rgb_3"+TIMESTAMP+".jpg", cv2.IMREAD_COLOR)
    rgb_cam3 = cv2.cvtColor(rgb_cam3, cv2.COLOR_BGR2RGB)
    depth_cam3 = np.load("data/"+FOLDER_NAME+"depth_3"+TIMESTAMP+".npy")
    depth_cam3 = gaussian_filter(depth_cam3, sigma, mode='constant')

    C_pcd_raw, C_pcd_full = load_pointclouds(rgb_cam3, depth_cam3, cam_intrinsics=CAM_INTRINSICS_4512)

    # A_pcd_raw.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # B_pcd_raw.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # test_correspondance(A_pcd_raw)

    A_pcd = downsample_pointclouds(A_pcd_raw)
    B_pcd = downsample_pointclouds(B_pcd_raw)
    C_pcd = downsample_pointclouds(C_pcd_raw)
    print("Loading time: ", time()-start_time)

    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw], width=WINDOW_WIDTH, height=WINDOW_HEIGHT) # plot A and B 
        o3d.visualization.draw_geometries([A_pcd,B_pcd], width=WINDOW_WIDTH, height=WINDOW_HEIGHT) # plot downsampled A and B 
    start_time = time()

    Ab_corr, Ba_corr, num_corrs = features_matching(A_pcd, B_pcd)
    Ac_corr, Ca_corr, _ = features_matching(A_pcd, C_pcd)
    Bc_corr, Cb_corr, _ = features_matching(B_pcd, C_pcd)
    print("Feature matching time: ", time()-start_time)

    # visualize the point clouds together with feature correspondences
    if VISUALIZE:
        draw_features_correspondences(A_pcd, B_pcd, Ab_corr, Ba_corr, num_corrs)
    start_time = time()

    # robust global registration using TEASER++
    Tab_teaser = global_registration(Ab_corr, Ba_corr)
    Tac_teaser = global_registration(Ac_corr, Ca_corr)
    Tbc_teaser = global_registration(Bc_corr, Cb_corr)
    print("TEASER++ registration time: ", time()-start_time)

    # Visualize the registration results
    if VISUALIZE:
        A_pcd_T_teaser = copy.deepcopy(A_pcd_raw).transform(Tab_teaser)
        o3d.visualization.draw_geometries([A_pcd_T_teaser,B_pcd_raw], width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    start_time = time()

    # local refinement using ICP
    Tab_icp = local_refinement(A_pcd, B_pcd, Tab_teaser)
    Tac_icp = local_refinement(A_pcd, C_pcd, Tac_teaser)
    Tbc_icp = local_refinement(B_pcd, C_pcd, Tbc_teaser)
    print("ICP refinment time: ", time()-start_time)
    print("Extrinsic matrix after registration: ", Tab_icp)
    r = R.from_matrix(Tab_icp[:3,:3])
    print("Rotation in euler angles: ", r.as_euler('xyz', degrees=True))

    # visualize the registration after ICP refinement
    if VISUALIZE_FINAL:
        A_pcd_T_icp = copy.deepcopy(A_pcd_raw).transform(Tab_icp)
        o3d.visualization.draw_geometries([A_pcd_T_icp,B_pcd_raw], width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    Ab_corr_inliers, Ba_corr_inliers = remove_correspondences_outliers(Ab_corr, Ba_corr, Tab_icp, dist_threshold=.03)
    Ac_corr_inliers, Ca_corr_inliers = remove_correspondences_outliers(Ac_corr, Ca_corr, Tac_icp, dist_threshold=.03)
    Bc_corr_inliers, Cb_corr_inliers = remove_correspondences_outliers(Bc_corr, Cb_corr, Tbc_icp, dist_threshold=.03)

    A_corr_inliers = get_common_correspondences(Ab_corr_inliers, Ac_corr_inliers, dist_threshold=.01)
    B_corr_inliers = get_common_correspondences(Ba_corr_inliers, Bc_corr_inliers, dist_threshold=.01)
    C_corr_inliers = get_common_correspondences(Ca_corr_inliers, Cb_corr_inliers, dist_threshold=.01)
    
    print(np.shape(A_corr_inliers)[0], " matched points cam1")
    print(np.shape(B_corr_inliers)[0], " matched points cam2")
    print(np.shape(C_corr_inliers)[0], " matched points cam3")

    A_px_coords = get_correspondences_pixel_coordinates(A_corr_inliers, A_pcd_full)
    B_px_coords = get_correspondences_pixel_coordinates(B_corr_inliers, B_pcd_full)
    C_px_coords = get_correspondences_pixel_coordinates(C_corr_inliers, C_pcd_full)
    
    color = [255,0,0]
    for px in A_px_coords:
        rgb_cam1[px] = color
    for px in B_px_coords:
        rgb_cam2[px] = color
    for px in C_px_coords:
        rgb_cam3[px] = color
    fig = plt.figure("Correspondences")
    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(rgb_cam1)
    ax.set_title('Cam1')
    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(rgb_cam2)
    ax.set_title('Cam2')
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(rgb_cam3)
    ax.set_title('Cam3')
    plt.show()
    
    m = len(A_px_coords)
    q21 = A_px_coords.pop(3*m//4)
    q22 = B_px_coords.pop(3*m//4)
    q01 = A_px_coords.pop(m//2)
    q02 = B_px_coords.pop(m//2)
    q11 = A_px_coords.pop(m//4)
    q12 = B_px_coords.pop(m//4)

    depthA = depth_cam1
    depthB = depth_cam2

    q0v = q02
    q1v = q12
    q2v = q22

    coeffs = []
    correct_points = []
    time_start = time()
    for i in range(m-3):
        # time_start = time()
        # print('--------------------------------------------------')
        # print("Processing the point ", i)
        # print('--------------------------------------------------')

        K1is = build_constraints_matrix(q01, q11, q21, depthA[q01], depthA[q11], depthA[q21], A_px_coords[i], depthA[A_px_coords[i]])
        K2is = build_constraints_matrix(q02, q12, q22, depthB[q02], depthB[q12], depthB[q22], B_px_coords[i], depthB[B_px_coords[i]])
        Kis = np.concatenate((K1is, K2is), axis=0) 
        _, D, V = np.linalg.svd(Kis)
        if D[-1] < 0.5:
            print("Residuals D: ", D)
            correct_points.append(i)
        coeffs.append(V[-1]/V[-1,-1])
        # print("Time: ", time()-time_start, "Structure coefficients: ", coeffs[-1])
    print("Time computing parameters: ", time()-time_start)
    print(correct_points)

    time_start = time()
    for i in correct_points:
        # Estimate the position of the current processed point in the new virtual view
        qv = (vv, uv) = B_px_coords[i]
        expected = np.array([uv, vv, depthB[q12]/depthB[q02], depthB[q22]/depthB[q02], depthB[B_px_coords[i]]/depthB[q02]])
        print("\nExpected u and v in the new view: ", expected)
        
        # Start the search from the position in reference view
        x0 = np.array([A_px_coords[i][1], A_px_coords[i][0], depthA[q11]/depthA[q01], depthA[q21]/depthA[q01], depthA[A_px_coords[i]]/depthA[q01]])
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