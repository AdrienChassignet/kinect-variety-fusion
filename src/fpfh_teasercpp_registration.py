import open3d as o3d
import cv2
import teaserpp_python
import numpy as np 
import copy
from helpers import *
import time
from scipy.spatial.transform import Rotation as R

VOXEL_SIZE = 0.025
VISUALIZE = False
VISUALIZE_FINAL = True

WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 720

FOLDER_NAME = ""
TIMESTAMP = "_20210430-075811"

MASTER_CAM_INTRINSICS = o3d.camera.PinholeCameraIntrinsic(1280,720,612.6449585,612.76092529,635.7354126,363.57376099)
SUB_CAM_INTRINSICS = o3d.camera.PinholeCameraIntrinsic(1280,720,611.85577393,611.68011475,638.63598633,369.67144775)

def load_pointclouds():
    # Load data

    rgb_cam1 = cv2.imread("data/"+FOLDER_NAME+"rgb_master"+TIMESTAMP+".jpg", cv2.IMREAD_COLOR)
    rgb_cam1 = cv2.cvtColor(rgb_cam1, cv2.COLOR_BGR2RGB)
    depth_cam1 = np.load("data/"+FOLDER_NAME+"depth_master"+TIMESTAMP+".npy")
    rgb_cam2 = cv2.imread("data/"+FOLDER_NAME+"rgb_sub"+TIMESTAMP+".jpg", cv2.IMREAD_COLOR)
    rgb_cam2 = cv2.cvtColor(rgb_cam2, cv2.COLOR_BGR2RGB)
    depth_cam2 = np.load("data/"+FOLDER_NAME+"depth_sub"+TIMESTAMP+".npy")

    rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_cam1), o3d.geometry.Image(depth_cam1), convert_rgb_to_intensity = False)
    A_pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, MASTER_CAM_INTRINSICS)
    # A_pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, o3d.camera.PinholeCameraIntrinsic(1280,720,504.21,504.11,319.30,320.30))
    rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_cam2), o3d.geometry.Image(depth_cam2), convert_rgb_to_intensity = False)
    B_pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, SUB_CAM_INTRINSICS)
    # B_pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, o3d.camera.PinholeCameraIntrinsic(1280,720,504.21,504.11,319.30,320.30))

    # A_pcd_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
    # B_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red

    return A_pcd_raw, B_pcd_raw

def downsample_pointclouds(pc1_raw, pc2_raw):
    # voxel downsample both clouds

    A_pcd = pc1_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    B_pcd = pc2_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    # A_pcd = pc1_raw.uniform_down_sample(10)
    # B_pcd = pc2_raw.uniform_down_sample(10)
    print("Nb points before removal: ", np.asarray(A_pcd.points).size)
    # Remove outliers that are far from other points
    A_pcd, _ = A_pcd.remove_statistical_outlier(100, .2)
    B_pcd, _ = B_pcd.remove_statistical_outlier(100, .2)
    print("Nb points after removal: ", np.asarray(A_pcd.points).size)

    return A_pcd, B_pcd

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
    print(f'FPFH generates {num_corrs} putative correspondences.')

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
    # Print the solution
    print("Solution is:", solution)
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

def main():

    start_time = time.time()

    A_pcd_raw, B_pcd_raw = load_pointclouds()

    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw], width=WINDOW_WIDTH, height=WINDOW_HEIGHT) # plot A and B 

    A_pcd, B_pcd = downsample_pointclouds(A_pcd_raw, B_pcd_raw)

    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd,B_pcd], width=WINDOW_WIDTH, height=WINDOW_HEIGHT) # plot downsampled A and B 

    print("Loading time: ", time.time()-start_time)
    start_time = time.time()

    A_corr, B_corr, num_corrs = features_matching(A_pcd, B_pcd)

    print("Feature matching time: ", time.time()-start_time, " // Found ", num_corrs, " correspondences")

    # visualize the point clouds together with feature correspondences
    if VISUALIZE:
        draw_features_correspondences(A_pcd, B_pcd, A_corr, B_corr, num_corrs)
    start_time = time.time()

    # robust global registration using TEASER++
    T_teaser = global_registration(A_corr, B_corr)

    print("TEASER++ registration time: ", time.time()-start_time)

    # Visualize the registration results
    if VISUALIZE:
        A_pcd_T_teaser = copy.deepcopy(A_pcd_raw).transform(T_teaser)
        o3d.visualization.draw_geometries([A_pcd_T_teaser,B_pcd_raw], width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    start_time = time.time()

    # local refinement using ICP
    T_icp = local_refinement(A_pcd, B_pcd, T_teaser)
    print(T_icp)
    r = R.from_matrix(T_icp[:3,:3])
    print(r.as_euler('xyz', degrees=True))

    print("ICP refinment time: ", time.time()-start_time)

    # visualize the registration after ICP refinement
    if VISUALIZE_FINAL:
        A_pcd_T_icp = copy.deepcopy(A_pcd_raw).transform(T_icp)
        o3d.visualization.draw_geometries([A_pcd_T_icp,B_pcd_raw], width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    matched_points = xyz2pcd(np.transpose(A_corr))
    matched_points.paint_uniform_color([0., 0., 1.])
    A_ref_points = o3d.geometry.PointCloud()
    B_ref_points = o3d.geometry.PointCloud()

    hull, tmp = matched_points.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))

    deltas = np.transpose(A_corr) - matched_points.get_center()
    centroid_idx = np.argmin(np.einsum('ij,ij->i', deltas, deltas))
    print(A_corr[:,centroid_idx], matched_points.get_center())
    matched_points.colors[centroid_idx] = [0., 1., 0.]
    A_ref_points.points.append(A_corr[:,centroid_idx])
    B_ref_points.points.append(B_corr[:,centroid_idx])

    for i in tmp:
        matched_points.colors[i] = [1,.5,.2]
    print(hull, tmp)
    hull.orient_triangles()
    hull.compute_vertex_normals()
    print(len(np.asarray(hull.vertex_normals)))
    print(hull_ls)
    hull.paint_uniform_color([1,0,0])


    A_corr_T_icp = copy.deepcopy(xyz2pcd(np.transpose(A_corr))).transform(T_icp)
    A_corr_T_icp = pcd2xyz(A_corr_T_icp)

    sel_low = False
    sel_up = False
    dist_thresh = .1
    for i, normal in enumerate(np.asarray(hull.vertex_normals)):
        # Remove outliers matches by distance in pointcloud
        dist = np.sqrt(np.sum((A_corr_T_icp[:,tmp[i]] - B_corr[:,tmp[i]])**2, axis=0))
        if dist < dist_thresh:
            print(dist)
            matched_points.colors[tmp[i]] = [0,1,0]
            if normal[0] > 0 and not sel_up:
                hull.vertex_colors[i] = [0,1,0]
                A_ref_points.points.append(A_corr[:,tmp[i]])
                B_ref_points.points.append(B_corr[:,tmp[i]])
                matched_points.colors[tmp[i]] = [0,1,1]
                sel_up = True
            elif normal[0] <= 0 and not sel_low:
                hull.vertex_colors[i] = [0,0,1]
                A_ref_points.points.append(A_corr[:,tmp[i]])
                B_ref_points.points.append(B_corr[:,tmp[i]])
                matched_points.colors[tmp[i]] = [1,0,1]
                sel_low = True
        if sel_low and sel_up:
            break

    o3d.visualization.draw_geometries([matched_points])

    A_ref_points.paint_uniform_color([0,0,1])
    B_ref_points.paint_uniform_color([0,0,1])
    vox_ref_points_A = o3d.geometry.VoxelGrid.create_from_point_cloud(A_ref_points, 0.08)
    o3d.visualization.draw_geometries([vox_ref_points_A, A_pcd_raw], width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    vox_ref_points_B = o3d.geometry.VoxelGrid.create_from_point_cloud(B_ref_points, 0.08)
    o3d.visualization.draw_geometries([vox_ref_points_B, B_pcd_raw], width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

    o3d.visualization.draw_geometries([vox_ref_points_A, vox_ref_points_B, A_pcd_raw, B_pcd_raw])

if __name__ == "__main__":
    main()
