import open3d as o3d
import cv2
import teaserpp_python
import numpy as np 
import copy
from helpers import *
import time

VOXEL_SIZE = 0.07
VISUALIZE = False
VISUALIZE_FINAL = False

def load_pointclouds():
    # Load data

    rgb_cam1 = cv2.imread("data/rgb_20210421-155829.jpg", cv2.IMREAD_COLOR)
    rgb_cam1 = cv2.cvtColor(rgb_cam1, cv2.COLOR_BGR2RGB)
    depth_cam1 = np.load("data/depth_20210421-155829.npy")
    rgb_cam2 = cv2.imread("data/rgb_20210421-155835.jpg", cv2.IMREAD_COLOR)
    rgb_cam2 = cv2.cvtColor(rgb_cam2, cv2.COLOR_BGR2RGB)
    depth_cam2 = np.load("data/depth_20210421-155835.npy")

    rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_cam1), o3d.geometry.Image(depth_cam1), convert_rgb_to_intensity = False)
    A_pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, o3d.camera.PinholeCameraIntrinsic(1280,720,612.35,612.13,639.86,363.73))
    # A_pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, o3d.camera.PinholeCameraIntrinsic(1280,720,504.21,504.11,319.30,320.30))
    rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(rgb_cam2), o3d.geometry.Image(depth_cam2), convert_rgb_to_intensity = False)
    B_pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, o3d.camera.PinholeCameraIntrinsic(1280,720,612.35,612.13,639.86,363.73))
    # B_pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, o3d.camera.PinholeCameraIntrinsic(1280,720,504.21,504.11,319.30,320.30))

    # A_pcd_raw.paint_uniform_color([0.0, 0.0, 1.0]) # show A_pcd in blue
    # B_pcd_raw.paint_uniform_color([1.0, 0.0, 0.0]) # show B_pcd in red

    return A_pcd_raw, B_pcd_raw

def downsample_pointclouds(pc1_raw, pc2_raw):
    # voxel downsample both clouds

    A_pcd = pc1_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    B_pcd = pc2_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    # A_pcd = pc1_raw.uniform_down_sample(20)
    # B_pcd = pc1_raw.uniform_down_sample(20)

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

    return A_corr, B_corr

def draw_features_correspondences(pc1, pc2, pc1_corr, pc2_corr):
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
    o3d.visualization.draw_geometries([pc1,pc2,line_set])

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
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

    return icp_sol.transformation

def main():

    start_time = time.time()

    A_pcd_raw, B_pcd_raw = load_pointclouds()

    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd_raw,B_pcd_raw]) # plot A and B 

    A_pcd, B_pcd = downsample_pointclouds(A_pcd_raw, B_pcd_raw)

    if VISUALIZE:
        o3d.visualization.draw_geometries([A_pcd,B_pcd]) # plot downsampled A and B 

    print("Loading time: ", time.time()-start_time)

    A_corr, B_corr = features_matching(A_pcd, B_pcd)

    print("Feature matching time: ", time.time()-start_time)

    # visualize the point clouds together with feature correspondences
    if VISUALIZE:
        draw_features_correspondences(A_pcd, B_pcd, A_corr, B_corr)

    # robust global registration using TEASER++
    T_teaser = global_registration(A_corr, B_corr)

    print("TEASER++ registration time: ", time.time()-start_time)

    # Visualize the registration results
    if VISUALIZE:
        A_pcd_T_teaser = copy.deepcopy(A_pcd_raw).transform(T_teaser)
        o3d.visualization.draw_geometries([A_pcd_T_teaser,B_pcd_raw])

    # local refinement using ICP
    T_icp = local_refinement(A_pcd, B_pcd, T_teaser)
    print(T_icp)

    print("ICP refinment time: ", time.time()-start_time)

    # visualize the registration after ICP refinement
    if VISUALIZE_FINAL:
        A_pcd_T_icp = copy.deepcopy(A_pcd_raw).transform(T_icp)
        o3d.visualization.draw_geometries([A_pcd_T_icp,B_pcd_raw])

    matched_points = xyz2pcd(np.transpose(A_corr))
    matched_points.paint_uniform_color([0., 0., 1.])
    hull, tmp = matched_points.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))

    deltas = np.transpose(A_corr) - matched_points.get_center()
    centroid_idx = np.argmin(np.einsum('ij,ij->i', deltas, deltas))
    print(A_corr[:,centroid_idx], matched_points.get_center())
    matched_points.colors[centroid_idx] = [0., 1., 0.]
    for i in tmp:
        matched_points.colors[i] = [1,.5,.2]
    print(hull, tmp)
    hull.orient_triangles()
    hull.compute_vertex_normals()
    print(len(np.asarray(hull.vertex_normals)))
    print(hull_ls)
    hull.paint_uniform_color([1,0,0])
    sel_low = False
    sel_up = False
    for i, normal in enumerate(np.asarray(hull.vertex_normals)):
        if normal[1] > 0 and not sel_up:
            hull.vertex_colors[i] = [0,1,0]
            sel_up = True
        elif normal[1] <= 0 and not sel_low:
            hull.vertex_colors[i] = [0,0,1]
            sel_low = True
        if sel_low and sel_up:
            break


    o3d.visualization.draw_geometries([matched_points, hull, hull_ls, A_pcd])

if __name__ == "__main__":
    main()
